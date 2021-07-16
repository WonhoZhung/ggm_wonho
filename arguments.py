import argparse
import utils


def train_args_parser():
    parser = argparse.ArgumentParser()                                              

    # DEFAULT SETTINGS
    parser.add_argument("--world_size", help="world size", type=int, default=4)        
    parser.add_argument("--distributed", help="distributed", action="store_true", \
            default=True)
    parser.add_argument("--autocast", help="autocast", action="store_true", \
            default=True)
    parser.add_argument("--num_workers", help="number of workers", type=int, \
            default=0)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=1)   

    # DIRECTORY SETTINGS
    parser.add_argument("--save_dir", help="save directory", type=str)
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str)
    
    # MODEL SETTINGS
    parser.add_argument("--num_layers", help="num layers", type=int)
    parser.add_argument("--max_num_nodes", help="max num nodes", type=int)
    parser.add_argument("--num_node_features", help="node features", type=int, \
            default=utils.NUM_ATOM_TYPES)
    parser.add_argument("--num_edge_features", help="edge features", type=int, \
            default=utils.NUM_BOND_TYPES)
    parser.add_argument("--num_node_hidden", help="node hidden features", \
            type=int)
    parser.add_argument("--num_edge_hidden", help="edge hidden features", \
            type=int)
    
    # TRAINING SETTINGS
    parser.add_argument("--num_epochs", help="num epochs", type=int)            
    parser.add_argument("--vae_coeff", help="vae_loss coeff", type=float, \
            default=1.0)
    parser.add_argument("--recon_coeff", help="recon_loss coeff", type=float, \
            default=1.0)
    parser.add_argument("--lr", help="lr", type=float, default=1e-5)
    parser.add_argument("--lr_decay", help="lr_decay", type=float, default=1.0)
    parser.add_argument("--weight_decay", help="weight_decay", type=float, \
            default=0.0)
    parser.add_argument("--restart_file", help="restart_file", type=str, \
            default=None)
    parser.add_argument("--save_every", help="save every n epochs", type=int, \
            default=1)
    parser.add_argument("--shuffle", help="shuffle", action="store_true")
    
    # RESULT SETTINGS
    parser.add_argument("--train_result_filename", help="train_result_filename", \
            type=str)
    parser.add_argument("--test_result_filename", help="test_result_filename", \
            type=str)
    
    args = parser.parse_args()
    return args

def generate_args_parser():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument("--num_workers", help="number of workers", type=int)
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--key_dir", help="key directory", type=str)
    parser.add_argument("--num_samples", help="num samples", type=int)
    
    parser.add_argument("--num_layers", help="num layers", type=int)
    parser.add_argument("--max_num_nodes", help="max num nodes", type=int)
    parser.add_argument("--max_add_edges", help="max add edges", type=int)
    parser.add_argument("--num_node_features", help="node features", type=int, \
            default=utils.NUM_ATOM_TYPES)
    parser.add_argument("--num_edge_features", help="edge features", type=int, \
            default=utils.NUM_BOND_TYPES)
    parser.add_argument("--num_node_hidden", help="node hidden features", type=int)
    parser.add_argument("--num_edge_hidden", help="edge hidden features", type=int)
    
    parser.add_argument("--restart_file", help="restart_file", type=str)
    
    args = parser.parse_args()
    return args
