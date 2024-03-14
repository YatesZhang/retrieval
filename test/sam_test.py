from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2 
import numpy as np
import torch
import pdb 
import sys 
sys.path.append("..")
import Flamingo
from Flamingo.models.modeling_sam import SAMImageTransforms
sam_checkpoint = "/root/yunzhi/checkpoint/sam/sam_vit_l_0b3195.pth"
model_type = "vit_l"

device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
predictor = SamPredictor(sam)
path = "/root/yunzhi/flamingo_retrieval/retrieval/Flamingo/images/yellow_bus.jpg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
input_point = np.array([[500, 375], [501,376], [502, 377], [503, 375], [504,376], [505, 377], [506, 377]])
input_label = np.array([1, 1, 1, 1, 1, 1, 1])
pdb.set_trace()
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

pdb.set_trace()
feature = predictor.features    # [1, 256, 256, 64]
# tensor(5611.8853, device='cuda:0')
# predictor.set_image(image)
# pdb.set_trace()

transforms = SAMImageTransforms(long_side_length=1024)
input_image = transforms(path)
# sam.image_encoder.eval()
input_image = input_image.to(device=device)
input_image = sam.image_encoder(input_image)
pdb.set_trace()
# ------------------------------------------------------------------------------
