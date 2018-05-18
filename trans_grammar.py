import nltk
import numpy as np
import six
import pdb

gram = """F -> '(' A 'r' S ')'
S -> '(' A 'r' S ')'
S -> '(' S '|' S ')'
S -> '(' S '.' S ')'
S -> A
A -> '0'
A -> '1'
A -> '2'
A -> '3'
A -> '4'
A -> '5'
A -> '6'
A -> '7'
A -> '8'
A -> '9'
A -> '10'
A -> '11'
Nothing -> None"""


GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()


all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0
#all_lhs.append(0)
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    #pdb.set_trace()
    masks[count] = is_in
    count = count + 1

index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)

