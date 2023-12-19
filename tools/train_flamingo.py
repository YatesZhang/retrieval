import math 
import deepspeed
# model config and dataset config: 
from Flamingo.config.baseline import dataset_config, model_config, workflows
# argparser:
from Flamingo.utils import parse_args 
# model:
from Flamingo.lora_tuning import create_model_and_transforms 
from Flamingo.models.batchprocessor import FlamingoBatchProcessor
# runner
from Flamingo.runner.deepspeed_runner import Runner as DeepSpeedRunner 
# DataLoader, DataSampler
from torch.utils.data import DataLoader, DistributedSampler
from Flamingo.datasets import InfiniteSampler, build_dataset
# optimizer and scheduler:
from transformers import AdamW
from transformers import get_scheduler
from Flamingo.utils.ds_utils import get_optimizer_grouped_parameters


def main():
    """ 
        build componets
    """
    deepspeed.init_distributed()
    # get DeepSpeed config and args 
    args = parse_args()

    # build model, image processor and tokenizer
    model, image_processor, tokenizer = create_model_and_transforms(
        **model_config
    )

    # build batch processor for model (Forward pass, return loss): 
    batch_processor = FlamingoBatchProcessor()

    # build dataloader:
    dataset = build_dataset(
        dataset_config=dataset_config,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )

    # dataloader: true batch_size is train_micro_batch_size_per_gpu
    train_dataloader = DataLoader(
        dataset,
        batch_size=ds_config['train_micro_batch_size_per_gpu'],   
        # num_workers=args.workers,    # 
        sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, small_lr=args.learning_rate_pretraining_components)    # default: learning rate

    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,    # cosine as default
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,    # 100 as default
        num_training_steps=sum([flow[1] for flow in workflows if flow[0] == 'train']) * num_update_steps_per_epoch,
    )
    
    runner = DeepSpeedRunner(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=None, 
        batch_processor=batch_processor,
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        workflows=workflows,
        args=args
    )
    runner.run() 
    
if __name__ == "__main__":
    main() 