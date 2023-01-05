import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import removeSkull
import feature_extraction
import classification_experiment

removeSkull.read_process_images('brain_tumor_dataset')
df_images = feature_extraction.createDataFrame()
df_features = feature_extraction.extract_features(df_images)


le = None
clf = SVC(kernel='poly')
#clf = DecisionTreeClassifier()
#clf = KNeighborsClassifier(n_neighbors=17)
#clf = GaussianNB()
X = df_features.drop(df_features['label'], axis=1)
Y = df_features['label']
classification_experiment.experiment(X, Y, clf, verbose=1)