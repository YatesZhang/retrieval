import deepspeed
from Flamingo.utils import parse_args 
from Flamingo.lora_tuning import create_model_and_transforms 
from Flamingo.runner.deepspeed_runner import Runner as DeepSpeedRunner 
from Flamingo.models.batchprocessor import FlamingoBatchProcessor

from torch.utils.data import DataLoader, DistributedSampler
from Flamingo.datasets import InfiniteSampler, build_dataset
# model config and dataset config: 
from Flamingo.config.baseline import dataset_config, model_config

def main():
    """ 
    deepspeed.init_distributed()
    """

    # get DeepSpeed config and args 
    args, ds_config = parse_args()

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
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )


if __name__ == "__main__":
    main() 