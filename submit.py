import os
import time
import sys


exp = "ggm_2"

for i in [1]:
    
    lines = f"""#!/bin/bash
#PBS -q batch
#PBS -N WH_{exp}.{i}
#PBS -l nodes=1:ppn=4:gpu
#PBS -l walltime=1000:00:00

source activate wonho

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

export OMP_NUM_THREADS=1
python -u train.py \
--save_dir save/{exp}.{i} \
--data_dir data/data/ \
--key_dir data/keys/ \
--num_layers 3 \
--max_num_nodes 30 \
--num_node_features 9 \
--num_edge_features 4 \
--num_node_hidden 128 \
--num_edge_hidden 128 \
--lr 1e-5 \
--lr_decay 0.99 \
--num_epochs 301 \
--save_every 1 \
--shuffle \
--train_result_filename results/result_{exp}.{i}_train.txt \
--test_result_filename results/result_{exp}.{i}_test.txt \
> log/{exp}.{i}.out 2> log/{exp}.{i}.err
    """
    
    with open("jobscript_submit.x", 'w') as w:
        w.writelines(lines)

    os.system("qsub jobscript_submit.x")

    time.sleep(10)
