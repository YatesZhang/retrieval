from Flamingo.utils.ds_utils import get_train_ds_config
import deepspeed 
from transformers import set_seed
from transformers import SchedulerType, get_scheduler
import torch 
import argparse 
import numpy as np 
import random 

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune Open Flamingo on a multi-modal task")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    # parser.add_argument(
    #     "--max_seq_len",
    #     type=int,
    #     default=4096,
    #     help="The maximum sequence length, note that image tokens are included.",
    # )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_pretraining_components",
        type=float,
        default=0,
        help=
        "Initial learning rate for pre-trained weight, e.g., embedding (after the potential warmup period) to use.",
    )
    
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=50,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=float,
        default=100,
        help="Number of steps (>1) or ratios (<=1) for the warmup in the lr scheduler.")
    
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    
    # deepspeed features
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=2,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    # BF16 is more stable than FP16 on training:
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help=
        "FP16 or BF16 precision. FP16 is recommended for typical use cases. BF16 is good for large models",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="../work_dir",
        help="work directory to save logs, checkpoints"
    )

    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.learning_rate_pretraining_components == 0.0:
        # if we do not provide special learning rate, mainly for embedding, the same lr is applied
        args.learning_rate_pretraining_components = args.learning_rate
    assert args.num_warmup_steps >= 0, "--num_warmup_steps must be >= 0"

    return args

def parse_args():
    """ 
        return: 
            1) DeepSpeed Config, 2) args from argparse
    """
    args = _parse_args()

    """  
    Load DeepSpeed Config
    assert: train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size
    """

    # DeepSpeed Config
    ds_config = get_train_ds_config(args, offload=False,
                                    stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps
    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    return args, ds_config 