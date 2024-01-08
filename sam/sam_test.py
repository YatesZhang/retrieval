import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

points_path = "/cfsdata2/cfsdata/workspace/yunzhi/flamingo_retrieval/retrieval/sam/data/1701153900630209_front_wide.npy"
img_path = "/cfsdata2/cfsdata/workspace/yunzhi/flamingo_retrieval/retrieval/sam/data/1701153900630209_front_wide_undist.jpg"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

if __name__ == '__main__':
    points = np.load(points_path)
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)