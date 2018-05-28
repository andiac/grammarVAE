from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from matplotlib import pyplot as plt

def to_np_cooc(x):
    return map(int, x.split())

x = pd.read_hdf('trans_latent.hdf5', 'table').values
print(x.shape)
y = np.array([int(line.strip()) for line in open('./data/big.label.ed', 'r')]) 
print(y.shape)
y2 = np.array([int(line.strip()) for line in open('./data/big.label2.ed', 'r')]) 
print(y2.shape)
y3 = np.array([to_np_cooc(line.strip()) for line in open('./data/big.cooclabel.ed', 'r')]) 
print(y3.shape)

originy2 = y2
num_classes_y = len(set(list(y)))
num_classes_y2 = len(set(list(y2)))
dim_y3 = y3.shape[1]
yy = y2
y = np_utils.to_categorical(y, num_classes_y)
y2 = np_utils.to_categorical(y2, num_classes_y2)

input_layer = Input(shape=(100,))
h = Dense(256, activation='relu', name="dense")(input_layer)
o1 = Dense(num_classes_y, activation='softmax')(h)
o2 = Dense(num_classes_y2, activation='softmax')(h)
o3 = Dense(100, activation='relu')(h)
o4 = Dense(dim_y3, activation = 'tanh')(h)
model = Model(input_layer, [o1, o2, o3, o4])

model.compile(loss='categorical_crossentropy',
              # optimizer=RMSprop(),
              optimizer=Adam(),
              metrics=['accuracy'],
              loss_weights=[0.0, 0.0, 0.0, 1.0])

history = model.fit(x, [y, y2, x, y3],
                    batch_size=32,
                    nb_epoch=30,
                    verbose=1,
                    validation_split=0.01)

featureExtractorModel = Model(model.input, model.get_layer('dense').output)

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score

z = featureExtractorModel.predict(x)
print("zshape", z.shape)
yy = KMeans(n_clusters=7, random_state=0).fit(z).labels_
print(normalized_mutual_info_score(originy2, yy))

pcaz = PCA(n_components=2).fit_transform(z)
plt.scatter(pcaz[:, 0], pcaz[:, 1], c=yy, s=1)
plt.colorbar()
plt.show()


pcaz = PCA(n_components=2).fit_transform(z)
plt.scatter(pcaz[:, 0], pcaz[:, 1], c=originy2, s=1)
plt.colorbar()
plt.show()
