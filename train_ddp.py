import os
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import arguments
from model import GGM
from dataset import GraphDataset, my_collate_fn
import utils

if torch.__version__ == "1.7.1":
    import torch.cuda.amp as amp
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.backends.cudnn as cudnn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler


def train(model, args, optimizer, data, train, device=None, scaler=None):
    model.train() if train else model.eval()

    i_batch = 0
    vae_losses, recon_losses, total_losses = [], [], []
    while True:
        sample = next(data, None)
        if sample is None:
            break
        scaff, whole = sample['scaff'], sample['whole']
        scaff.graph_to_device(device)
        whole.graph_to_device(device)
        
        if args.autocast:
            with amp.autocast():
                scaff_save, vae_loss, recon_loss = \
                        model(scaff, whole, train=train)
                total_loss = vae_loss + recon_loss
        else:
            scaff_save, vae_loss, recon_loss = \
                    model(scaff, whole, train=train)
            total_loss = vae_loss + recon_loss

        if train:
            optimizer.zero_grad()
            if args.autocast:
                scaler.scale(total_loss).backward(retain_graph=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward(retain_graph=True)
                optimizer.step()

        vae_losses.append(vae_loss.data.cpu().numpy())
        recon_losses.append(recon_loss.data.cpu().numpy())
        total_losses.append(total_loss.data.cpu().numpy())

    vae_losses = np.mean(np.array(vae_losses))
    recon_losses = np.mean(np.array(recon_losses))
    total_losses = np.mean(np.array(total_losses))

    return vae_losses, recon_losses, total_losses


def main_worker(gpu, ngpus_per_node, args):

    ############ Distributed Data Parallel #############
    # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    rank = gpu
    print("Rank:", rank, flush=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "2021"
    dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=args.world_size
    )
    print("######## Finished Setting DDP ########", flush=True)
    ####################################################

    # Path
    save_dir = utils.get_abs_path(args.save_dir)
    data_dir = utils.get_abs_path(args.data_dir)
    if args.restart_file:
        restart_file = utils.get_abs_path(args.restart_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Dataloader
    train_dataset = GraphDataset(args, mode='train')
    #test_dataset = GraphDataset(args, mode='test')

    ############ Distributed Data Parallel #############
    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank
    )
    #test_sampler = DistributedSampler(
    #        test_dataset,
    #        num_replicas=args.world_size,
    #        rank=rank,
    #        shuffle=False
    #)
    ####################################################
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=my_collate_fn,
            shuffle=False,
            sampler=train_sampler
    )
    #test_dataloader = DataLoader(
    #        test_dataset,
    #        args.batch_size,
    #        num_workers=0,
    #        pin_memory=True,
    #        collate_fn=my_collate_fn,
    #        shuffle=False,
    #        sampler=test_sampler
    #)
    N_TRAIN_DATA = len(train_dataset)
    if not args.restart_file and rank == 0:
        print("Train dataset length: ", N_TRAIN_DATA, flush=True)
        #print("Test dataset length: ", len(test_dataset))
        print("######## Finished Loading Datasets ########", flush=True)

    # Model initialize
    model = GGM(args)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    ############ Distributed Data Parallel #############
    # Wrap the model
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    cudnn.benchmark = True
    ####################################################

    if not args.restart_file and rank == 0:
        print("Number of Parameters: ", \
              sum(p.numel() for p in model.parameters() if p.requires_grad), \
              flush=True)
        print("####### Finished Loading Model #######", flush=True)

    # Optimizer
    optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
    )

    # Scaler (AMP)
    if args.autocast:
        scaler = amp.GradScaler()

    # Train
    if args.restart_file:
        start_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        if epoch == 0 and rank == 0:
            print(f"EPOCH\t|\tVAE\t|\tRECN.\t|\tTOTAL\t|\tTIME/DATA", flush=True)

        train_data = iter(train_dataloader)
        #test_data = iter(test_dataloader)

        st = time.time()

        train_vae_losses, train_recon_losses, train_total_losses = \
                train(
                        model=model, 
                        args=args,
                        optimizer=optimizer,
                        data=train_data,
                        train=True,
                        device=gpu,
                        scaler=scaler
                )

        et = time.time()
        
        if rank == 0:
            print(f"{epoch}\t|\t{train_vae_losses:.3f}\t|\t" + \
                  f"{train_recon_losses:.3f}\t|\t{train_total_losses:.3f}\t|\t" + \
                  f"{(et - st)/N_TRAIN_DATA:.2f}", flush=True)

        name = os.path.join(save_dir, f"save_{epoch}.pt")
        save_every = 1 if not args.save_every else args.save_every
        if epoch % save_every == 0 and rank == 0:
            torch.save(model.state_dict(), name)

        lr = args.lr * ((args.lr_decay)**epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def main():
    args = arguments.train_args_parser()
    d = vars(args)
    print("####################################################")
    for a in d: print(a, "=", d[a])
    print("####################################################")

    # Seed
    torch.manual_seed(0)
    np.random.seed(0)
    cudnn.deterministic = True

    if args.distributed:
        mp.spawn(
                main_worker, 
                nprocs=args.world_size, 
                args=(args.world_size, args,),
        )
    else:
        main_worker(0, args.world_size, args)


if __name__ == "__main__":
    
    main()

