from __future__ import division
import sys

import trans_vae
import pandas as pd

import numpy as np
from numpy import sin, exp, cos
from matplotlib import pyplot as plt

import argparse
import metric

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices = ["2db", "big"], default="2db")
parser.add_argument("--weight_path", type=str, default="./trained_2db.hdf5")
args = parser.parse_args()

grammar_weights = args.weight_path
grammar_model = trans_vae.EquationGrammarModel(grammar_weights, latent_rep_size=25)
trans_file = open("./data/" + args.dataset + ".cfg", "r")
# trans_file = open("./data/2db.cfg", "r")
structure_label_file = open("./data/" + args.dataset + ".label", "r")
function_label_file = open("./data/" + args.dataset + ".label2", "r")
cooc_label_file = open("./data/" + args.dataset + ".cooclabel", "r")

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

# ed: eliminate duplicate
structure_label_file = open("./data/" + args.dataset + ".label.ed", "w")
function_label_file = open("./data/" + args.dataset + ".label2.ed", "w")
cooc_label_file = open("./data/"+ args.dataset + ".cooclabel.ed", "w")

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
# print("heihei")
# pcax = PCA(n_components=2).fit_transform(x)
# plt.scatter(pcax[:, 0], pcax[:, 1], c=function_labels, s=5, cmap="Set1")
# plt.colorbar()
# # for i, txt in enumerate(list(set(trans))):
# #     plt.annotate(txt, (x[i, 0], x[i, 1]), fontsize=6)
# print("haha")
# plt.show()

tsnex = TSNE(n_components=2, perplexity=30.0).fit_transform(x)
plt.scatter(tsnex[:, 0], tsnex[:, 1], c=function_labels, s=5, cmap=plt.cm.get_cmap('Set1', len(set(function_labels))))
plt.colorbar()
plt.show()

plt.scatter(tsnex[:, 0], tsnex[:, 1], c=structure_labels, s=5, cmap=plt.cm.get_cmap('Set1', len(set(structure_labels))))
plt.colorbar()
plt.show()

# yy = KMeans(n_clusters=9, random_state=0).fit(x).labels_
# plt.scatter(tsnex[:, 0], tsnex[:, 1], c=yy, s=5, cmap="Set1")
# plt.colorbar()
# plt.show()

from sklearn.metrics.cluster import normalized_mutual_info_score
# print(normalized_mutual_info_score(function_labels, yy))

# for e in [0.2, 0.205, 0.21, 0.215, 0.22]:
#     y = DBSCAN(eps=e).fit_predict(x)
#     print(y)
#     plt.scatter(tsnex[:, 0], tsnex[:, 1], c=y, s=1, cmap="Set1")
#     plt.colorbar()
#     plt.show()
#     print(normalized_mutual_info_score(function_labels, y))

def gao(labels):
    n_classes = max(labels) + 1
    best_nmi = 0.0
    best_k = None
    best_purity = None
    best_f1 = None
    for i in range(n_classes - 1, n_classes + 3):
        kmeansy = KMeans(n_clusters=i, random_state=0).fit(x).labels_
        nmi = normalized_mutual_info_score(labels, kmeansy)
        purity = metric.purity_score(kmeansy, labels)
        f1 = metric.f1_score(kmeansy, labels)
        if nmi > best_nmi:
            best_nmi = nmi
            best_k = i
            best_purity = purity
            best_f1 = f1
    print("best kmeans (nmi, k, purity, f1): ", best_nmi, best_k, best_purity, best_f1)

    s_labels = labels
    # best dbscan
    for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        dbscany = DBSCAN(eps=e).fit_predict(x)
        nmi = normalized_mutual_info_score(s_labels, dbscany)
        purity = metric.purity_score(dbscany, s_labels)
        f1 = metric.f1_score(dbscany, s_labels)
        #print(e, nmi)
        print("nmi, k, purity, p, r, f1: ", nmi, len(set(dbscany)), purity, f1)

gao(function_labels)
gao(structure_labels)
