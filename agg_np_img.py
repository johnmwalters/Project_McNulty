import os
from tqdm import tqdm # smart progress bar
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc_params
import pandas as pd
import numpy as np

#skimage features and filters
from skimage.feature import daisy
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
 
# get our styles
mpl_default = rc_params()
 
import seaborn as sns
 
# for ipython notebook %matplotlib inline
plt.rcParams = mpl_default

# BEES colormap! Black -> Yellow
CMAP = [(0,0,0),
        (22,0,0),
        (43,0,0),
        (77,14,0),
        (149,68,0),
        (220,123,0),
        (255,165,0),
        (255,192,0),
        (255,220,0),
        (255,235,0),
        (255,245,0),
        (255,255,0)]

bees_cm = mpl.colors.ListedColormap(np.array(CMAP)/255.)

# load the labels using pandas
labels = pd.read_csv("data/train_labels.csv",index_col=0)

submission_format = pd.read_csv("data/SubmissionFormat.csv",index_col=0)

#import functions

from get_image import get_image
from extract_rgb_info import extract_rgb_info
from preprocess import preprocess
from create_feature_matrix import create_feature_matrix

# turn those images into features!
bees_features = create_feature_matrix(labels)

# save in case we need to load later; creating this takes
# a little while and it ends up being ~3GBs
np.save("bees_features.npy", bees_features)