import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def show_pred_with_gt(imgs, preds, gts):
    """ 
        show prediction with ground truth 
    """
    _imgs = []
    for img in imgs:
        if isinstance(img, Image.Image):
            img = np.array(img).astype('uint8')
        else:
            img = img.astype('uint8')
        _imgs.append(img)
    imgs = _imgs
    
    plt.figure(figsize=(25, 25))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        actual = gts[i]
        plt.xlabel('GT:{}'.format(actual), color = 'g', fontsize=14)
        if preds is not None:
            prediction = preds[i]
            plt.ylabel('Pred:{}'.format(prediction), color = 'b', fontsize=14)
        plt.imshow(imgs[i])
    plt.show()
    return 