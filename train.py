import os
import utils
import time
import dataset
import numpy as np

from arguments import train_args_parser
from tqdm import tqdm

# Set args
args = train_args_parser()
print(args)

import torch

cmd = utils.set_cuda_visible_device(ngpus=1)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

from model import GGM


def run(model, dataset, train=True):
    
    model.train() if train else model.eval()
    vae_losses, recon_losses, total_losses = [], [], []
    
    for i in tqdm(range(len(dataset))):
        model.zero_grad()

        sample = dataset.__getitem__(i)
        scaff, whole = sample['scaff'], sample['whole']
        scaff.graph_to_device(device)
        whole.graph_to_device(device)

        with torch.autograd.set_detect_anomaly(True):
            scaff_save, vae_loss, recon_loss = model(scaff, whole, train)
            total_loss = vae_loss + recon_loss

            if train:
                total_loss.backward(retain_graph=True)
                optimizer.step()

        vae_losses.append(vae_loss.data.cpu().numpy())
        recon_losses.append(recon_loss.data.cpu().numpy())
        total_losses.append(total_loss.data.cpu().numpy())

    vae_losses = np.mean(np.array(vae_losses))
    recon_losses = np.mean(np.array(recon_losses))
    total_losses = np.mean(np.array(total_losses))

    return vae_losses, recon_losses, total_losses

# Make directory for save files
os.makedirs(args.save_dir, exist_ok=True)

# Model
model = GGM(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file=args.restart_file)

if not args.restart_file:
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() \
            if p.requires_grad))

# Optimizer and loss
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
)

# Dataset
graph_training_dataset = dataset.GraphDataset(args)

if args.restart_file:
    start_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
else:
    start_epoch = 0

for epoch in range(start_epoch, args.num_epochs + 1):
    st = time.time()

    train_vae_losses, train_recon_losses, train_total_losses = \
            run(model, graph_training_dataset, True)

    if epoch == start_epoch:
        print(f"EPOCH\t|\tVAE\t|\tRECN.\t|\tTOTAL")
    print(f"{epoch}\t|\t{train_vae_losses:.3f}\t|\t" + \
          f"{train_recon_losses:.3f}\t|\t{train_total_losses:.3f}")

    name = os.path.join(args.save_dir, 'save_' + str(epoch) + '.pt')
    save_every = 1 if not args.save_every else args.save_every
    if epoch % save_every == 0:
        torch.save(model.state_dict(), name)

    lr = args.lr * ((args.lr_decay) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
