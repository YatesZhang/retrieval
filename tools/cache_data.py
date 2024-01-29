try:
    import Flamingo
except ModuleNotFoundError:
    import sys
    sys.path.append("..")
    import Flamingo
import pdb
import os 
from Flamingo.models.modeling_clip import get_clip_vision_encoder_and_processor
from Flamingo.models.batchprocessor import CLIPBatchProcessor
from Flamingo.runner.clip_cacher import CacheCLIPOutRunner as Runner
from Flamingo.datasets.participants_property import ParticipantsProperty
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
""" 
CUDA_VISIBLE_DEVICES=0,2,3,4,5,7 python cache_data.py
CUDA_VISIBLE_DEVICES=7 python cache_data.py
ps -ef | grep defunct | more
ps -ef | grep python 
fuser -v /dev/nvidia*
pkill -u $USER python
"""
if __name__ == "__main__":
    vision_encoder, image_processor = get_clip_vision_encoder_and_processor()
    vision_encoder.share_memory()
    batch_processor = CLIPBatchProcessor(vision_encoder=vision_encoder,
                                        image_processor=image_processor)
    """ 
        config:
    """
    
    annFile = "/root/datasets/participant_property/labels/train/train_coco.json"
    imgs_dir = "/root/datasets/participant_property/images"
    cache_dir = "/root/datasets/participants_property_clip"
    
    
    # create dataset:
    dataset = ParticipantsProperty(annFile=annFile, imgs_dir=imgs_dir)
    pdb.set_trace()
    # create runner:
    batch_size_per_gpu=16
    runner = Runner(
        batch_processor=batch_processor,
        cache_dir=cache_dir,
        dataset=dataset,
        batch_size_per_gpu=batch_size_per_gpu,
        # mp_size=4,   # model parallel size
        split='train'
    )
    # pdb.set_trace()
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        # mp.spawn(torch.multiprocessing.exit, nprocs=torch.cuda.device_count())
        # dist.destroy_process_group()  
        torch.cuda.empty_cache()
        os.system("kill $(ps aux | grep cache_data.py | grep -v grep | awk '{print $2}') ")
        print("killed Signal")