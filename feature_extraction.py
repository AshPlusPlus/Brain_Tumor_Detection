
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import tensorflow as tf
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from keras import applications
from keras import preprocessing
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
import os
from PIL import Image
import removeSkull


def createDataFrame():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()


    images = []
    yes_images = []
    no_images = []
    valid_images = [".jpg",".jpeg"]


    for f in os.listdir('removedSkull/yes/'):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = (keras.preprocessing.image.load_img(os.path.join('removedSkull/yes/', f), target_size=(225, 225)))
        image = img_to_array(image)
        yes_images.append(image)

    yes_label = [1] * len(yes_images)

    df1['image'] = yes_images
    df1['label'] = yes_label

    for f in os.listdir('removedSkull/no/'):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = (keras.preprocessing.image.load_img(os.path.join('removedSkull/no/',f), target_size=(225, 225)))
        image = img_to_array(image)
        no_images.append(image)

    no_label = [0] * len(no_images)

    df2['image'] = no_images
    df2['label'] = no_label
    df1 = df1.append(df2)
    x = df1.shape
    df1 = df1.sample(frac=1)
    return df1

def extract_features(df_images):
    images = df_images['image'].values.tolist()
    labels = df_images['label'].values.tolist()
    done = 0

    print("[INFO] loading network...")
    model = applications.VGG19(weights="imagenet", include_top=False)
    for image in images:
        image = tf.expand_dims(image, axis=0)
        image = applications.vgg19.preprocess_input(image)
        features = model.predict(image)
        features = features.reshape((features.shape[0], -1))
        if done == 0:
            images_features = features
            done = 1
        else:
            images_features = np.concatenate((images_features, features), axis=0)

    images_features = np.asarray(images_features)
    df_features = pd.DataFrame(images_features)
    df_features['label'] = labels
    return df_features








