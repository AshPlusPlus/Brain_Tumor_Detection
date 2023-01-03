from sklearn.preprocessing import LabelEncoder
from keras import applications
from keras import preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import pickle
import random
import os

print("[INFO] loading network...")
model = applications.VGG16(weights="imagenet", include_top=False)
le = None


