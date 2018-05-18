from __future__ import division
import sys

import equation_vae

import numpy as np
from numpy import sin, exp, cos
from matplotlib import pyplot as plt
import pdb

grammar_weights = "./eq_vae_grammar_h100_c234_L25_E50_batchB.hdf5"
grammar_model = equation_vae.EquationGrammarModel(grammar_weights, latent_rep_size=25)
equation_file = open("./data/equation2_15_dataset.txt", "r")

CASE = 10000
eq = []

for line in equation_file:
    eq.append(line.strip())
equation_file.close()

eq = eq[0:CASE]

z = grammar_model.encode(eq)
z2 = grammar_model.get_dense(eq)

for i, s in enumerate(grammar_model.decode(z)):
    print(eq[i], s)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# tsne = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=itera, method='exact')
x = z2
y = range(CASE)
print("heihei")
x = TSNE(n_components=2, perplexity=30.0).fit_transform(x)
plt.scatter(x[:, 0], x[:, 1], c=y, s=1)
print("haha")
plt.show()
