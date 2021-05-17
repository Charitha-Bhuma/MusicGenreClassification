import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
from keras.models import Sequential

header = 'filename zero_crossing_rate_mean zero_crossing_rate_median zero_crossing_rate_std spectral_centroid_mean spectral_centroid_median spectral_centroid_std spectral_contrast_mean spectral_contrast_median spectral_contrast_std spectral_bandwidth_mean spectral_bandwidth_median spectral_bandwidth_std spectral_rolloff_mean spectral_rolloff_median spectral_rolloff_std'
for i in range(1, 21):
    header += f' mfcc_mean{i}'
    header += f' mfcc_median{i}'
    header += f' mfcc_std{i}'
header += ' label'
header = header.split()

file = open('dataset.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
        songname = f'./drive/My Drive/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        zcr = librosa.feature.zero_crossing_rate(y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(zcr)} {np.median(zcr)} {np.std(zcr)} {np.mean(spec_cent)} {np.median(spec_cent)} {np.std(spec_cent)} {np.mean(spec_con)} {np.median(spec_con)} {np.std(spec_con)} {np.mean(spec_bw)} {np.median(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.median(rolloff)} {np.std(rolloff)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
            to_append += f' {np.median(e)}'
            to_append += f' {np.std(e)}'
        to_append += f' {g}'
        file = open('dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

data = pd.read_csv('dataset.csv')
data.head()
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier  
knn_classifier= KNeighborsClassifier(n_neighbors=3, metric='manhattan', p=2 )  
knn_classifier.fit(x_train, y_train)

knn_y_pred= knn_classifier.predict(x_test)  

c = 0
print(len(knn_y_pred))
for i in range(len(knn_y_pred)) :
  if(knn_y_pred[i] == y_test[i]) : 
    c += 1
c

from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, knn_y_pred) 

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, knn_y_pred)
acc
