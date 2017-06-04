import importlib
import utils2; importlib.reload(utils2)
from utils2 import *
import numpy as np

# Setup for training images
PATH = '/data/CityScrapes_dataset/cityscapes/leftImg8bit/'
frames_path = PATH+'all_train_images/'
labels_path = PATH+'all_train_gt/'


# fnames: file name of each training image
fnames = glob.glob(frames_path+'*.png')
lnames = glob.glob(labels_path+'*.png')
#lnames = [labels_path+os.path.basename(fn) for fn in fnames]
img_sz = (2048, 1024)

def open_image(fn): return np.array(Image.open(fn))

imgs = np.stack([open_image(fn) for fn in fnames[:100]])
labels = np.stack([open_image(fn) for fn in lnames[:100]])

imgs = imgs/255.

n,r,c,ch = imgs.shape
imgs-=0.4
imgs/=0.3

print('Save {} imgs to npz filew'.format(imgs.shape[0]))
np.save('imgs.npz', imgs)
print('Saved...')