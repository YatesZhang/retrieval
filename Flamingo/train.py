import argparse
import deepspeed
deepspeed.init_distributed()

def parse_arguments():
    parser = argparse.ArgumentParser(description="DeepSpeed training")



def train(args, model, dataset):
    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=deepspeed_config,
    )