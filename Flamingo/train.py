import argparse
import deepspeed
from rich import print
import pdb 

deepspeed.init_distributed()

def parse_arguments():
    # init argparse:
    parser = argparse.ArgumentParser(description="DeepSpeed training")
    
    # Data.
    # Cuda.
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')

    # Train.
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    """ 
    python train.py \
    --local_rank 0 \
    --deepspeed_config ds_config.json
    """
    # get arguments:
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments() 
    print("args: ")
    print(args) 
    pdb.set_trace()
    print(args.deepspeed_config)

if __name__ == '__main__':
    main()