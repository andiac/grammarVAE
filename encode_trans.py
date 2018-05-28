from __future__ import division
import sys

import trans_vae
import pandas as pd

import numpy as np
from numpy import sin, exp, cos
from matplotlib import pyplot as plt

grammar_weights = "./trans_vae_grammar_h100_c234_L25_E50_batchB.hdf5"
grammar_model = trans_vae.EquationGrammarModel(grammar_weights, latent_rep_size=25)
trans_file = open("./data/big.cfg", "r")
# trans_file = open("./data/2db.cfg", "r")
structure_label_file = open("./data/big.label", "r")
function_label_file = open("./data/big.label2", "r")
cooc_label_file = open("./data/big.cooclabel", "r")

trans = []
structure_labels = []
function_labels = []
cooc_labels = []
for line in trans_file:
    trans.append(line.strip())
for line in structure_label_file:
    structure_labels.append(int(line.strip()))
for line in function_label_file:
    function_labels.append(int(line.strip()))
for line in cooc_label_file:
    cooc_labels.append(line)
trans_file.close()
structure_label_file.close()
function_label_file.close()
cooc_label_file.close()

# eliminate duplicate
trans, structure_labels, function_labels, cooc_labels = zip(*list(set(zip(trans, structure_labels, function_labels, cooc_labels))))
trans = list(trans)
structure_label_file = list(structure_labels)
function_labels = list(function_labels)
cooc_labels = list(cooc_labels)

structure_label_file = open("./data/big.label.ed", "w")
function_label_file = open("./data/big.label2.ed", "w")
cooc_label_file = open("./data/big.cooclabel.ed", "w")

structure_label_file.writelines(map(lambda x: str(x) + "\n", structure_labels))
function_label_file.writelines(map(lambda x: str(x) + "\n", function_labels))
cooc_label_file.writelines(cooc_labels)

structure_label_file.close()
function_label_file.close()
cooc_label_file.close()

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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# tsne = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=itera, method='exact')
x = z2
# y = KMeans(n_clusters=7, random_state=0).fit(x).labels_
print("heihei")
pcax = PCA(n_components=2).fit_transform(x)
plt.scatter(pcax[:, 0], pcax[:, 1], c=function_labels, s=1, cmap="Set1")
plt.colorbar()
# for i, txt in enumerate(list(set(trans))):
#     plt.annotate(txt, (x[i, 0], x[i, 1]), fontsize=6)
print("haha")
plt.show()


tsnex = TSNE(n_components=2, perplexity=30.0).fit_transform(x)
plt.scatter(tsnex[:, 0], tsnex[:, 1], c=function_labels, s=1, cmap="Set1")
plt.colorbar()
plt.show()

yy = KMeans(n_clusters=9, random_state=0).fit(x).labels_
plt.scatter(tsnex[:, 0], tsnex[:, 1], c=yy, s=1, cmap="Set1")
plt.colorbar()
plt.show()

from sklearn.metrics.cluster import normalized_mutual_info_score
print(normalized_mutual_info_score(function_labels, yy))

for e in [0.2, 0.205, 0.21, 0.215, 0.22]:
    y = DBSCAN(eps=e).fit_predict(x)
    print(y)
    plt.scatter(tsnex[:, 0], tsnex[:, 1], c=y, s=1, cmap="Set1")
    plt.colorbar()
    plt.show()
    print(normalized_mutual_info_score(function_labels, y))
