import os
import utils
import time
import dataset
import numpy as np

from arguments import generate_args_parser
from tqdm import tqdm

# Set args
args = generate_args_parser()
print(args)

import torch

cmd = utils.set_cuda_visible_device(ngpus=1)
os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

from model import GGM


def run(model, dataset, train=False):
    
    model.train() if train else model.eval()
    
    for i in tqdm(range(len(dataset))):
        model.zero_grad()

        sample = dataset.__getitem__(i)
        scaff, whole = sample['scaff'], sample['whole']
        scaff.graph_to_device(device)
        whole.graph_to_device(device)

        samples = []

        for _ in range(args.num_samples):

            scaff_save = model(scaff, train=train)
            samples.append(scaff_save)

    return samples

# Make directory for save files
os.makedirs(args.save_dir, exist_ok=True)

# Model
model = GGM(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file=args.restart_file)

if not args.restart_file:
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() \
            if p.requires_grad))

# Dataset
graph_training_dataset = dataset.GraphDataset(args)

if args.restart_file:
    start_epoch = int(args.restart_file.split("_")[-1].split(".")[0])
else:
    start_epoch = 0

for epoch in range(start_epoch, args.num_epochs + 1):
    st = time.time()

    samples = run(model, graph_training_dataset, True)

    et = time.time()

