from accelerate import Accelerator
from accelerate.utils import gather_object
try:
    import Flamingo 
except:
    import sys
    sys.path.append('..')

from Flamingo.datasets import ParticipantsProperty 
import pdb
accelerator = Accelerator()

import torch.multiprocessing as mp
from model import MyModel

# def train(model):
#     # Construct data_loader, optimizer, etc.
#     for data, labels in data_loader:
#         optimizer.zero_grad()
#         loss_fn(model(data), labels).backward()
#         optimizer.step()  # This will update the shared parameters

# if __name__ == '__main__':
#     num_processes = 4
#     model = MyModel()
#     # NOTE: this is required for the ``fork`` method to work
#     model.share_memory()
#     processes = []
#     for rank in range(num_processes):
#         p = mp.Process(target=train, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
if __name__ == '__main__':
    annFile = "/root/datasets/participant_property/participant_property/labels/val/valid_coco.json"
    imgs_dir = "/root/datasets/participant_property/participant_property/images"
    dataset = ParticipantsProperty(annFile=annFile, imgs_dir=imgs_dir)
    with accelerator.split_between_processes(dataset) as acc_dataset:
        for data in acc_dataset:
            print(data)
            pdb.set_trace()

