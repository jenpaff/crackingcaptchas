# crackingcaptchas
This Code was developed in the course of my Research Project on Cracking CAPTCHAs.

**PRETRAINING_AUTOENCODER
-extractFeatures.py
-samples.mat
-test_extractFeatures.py

**FINETUNING_CNN
-convnet_captchas.py
-gui_captchas.py
-training_set.mat
-test_set.mat
-test_conv.py
-Autoencoder_results/mean.npy
-Autoencoder_results/opt_parameter_vector.npy
-Autoencoder_results/zca_white.npy


############## To run the CAPTCHA GUI: 
python gui_catpcha.py

The following libraries are imported: 
from Tkinter import *
from PIL import Image as img
from PIL import ImageTk as imgtk
import convnet_captchas
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time


########### To run the Convolutional Neural Network

python convnet_captchas.py

The following libraries are imported: 
import numpy as np
import scipy.io
from scipy import signal
from PIL import Image as img
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import random as rnd
import time

########## To run the Sparse Autoencoder for Pre-training 

python extract_features.py

The following libraries are imported: 
import numpy as np
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt

*Note that the Sparse Autoencoder runs very slowly
