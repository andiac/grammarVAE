from __future__ import division
import sys

import trans_vae
import pandas as pd

import numpy as np
from numpy import sin, exp, cos
from matplotlib import pyplot as plt

grammar_weights = "./trans_vae_grammar_h100_c234_L25_E50_batchB.hdf5"
grammar_model = trans_vae.EquationGrammarModel(grammar_weights, latent_rep_size=25)
# trans_file = open("./data/big.cfg", "r")
trans_file = open("./data/2db.cfg", "r")

trans = []

for line in trans_file:
    trans.append(line.strip())
trans_file.close()

trans = list(set(trans))

CASE = len(trans)

trans = trans[0:CASE]

z = grammar_model.encode(trans)
z2 = grammar_model.get_dense(trans)

df = pd.DataFrame(z2)
df.to_hdf('trans_latent.hdf5', 'table')

count = 0
correct = 0
for i, s in enumerate(grammar_model.decode(z)):
    print(trans[i], s)
    if ''.join(trans[i].strip().split()) == s.strip():
        correct += 1
    count += 1

print(correct / count)

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# tsne = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=itera, method='exact')
x = z2
y = KMeans(n_clusters=8, random_state=0).fit(x).labels_
print("heihei")
#x = TSNE(n_components=2, perplexity=30.0).fit_transform(x)
x = PCA(n_components=2).fit_transform(x)
plt.scatter(x[:, 0], x[:, 1], c=y, s=1)
plt.colorbar()
for i, txt in enumerate(list(set(trans))):
    plt.annotate(txt, (x[i, 0], x[i, 1]))
print("haha")
plt.show()
