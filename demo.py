import sys
from tqdm import tqdm
from glob import glob
from utils import Node, traverse_label, traverse
import pickle
import os
from joblib import Parallel, delayed
from collections import Counter
import re
from os.path import abspath
import nltk
# import os
# from utils import Node, traverse_label, traverse
# def parse(path):
#     with open(path, "r") as f:
#         num_objects = f.readline()
#         nodes = [Node(num=i, children=[]) for i in range(int(num_objects))]   # 得到有多少个结点
#         for i in range(int(num_objects)):
#             label = " ".join(f.readline().split(" ")[1:])[:-1]
#             nodes[i].label = label
#         while 1:
#             line = f.readline()
#             if line == "\n":
#                 break
#             p, c = map(int, line.split(" "))
#             nodes[p].children.append(nodes[c])
#             nodes[c].parent = nodes[p]
#         nl = f.readline()[:-1]
#         print(nl)
#     return nodes[0], nl
#
#
# node,nl = parse("/Users/chao/Desktop/code_dataset/demo/0")
# print(node)
# print(nl)

labels = ["a","b","c","ccc","aaaa","ddd","aaaa"]

ids = Counter(
    [y for y in [x for x in tqdm(
        labels, "collect identifiers")] if y is not None])
print(ids)
ids_list = [x[0] for x in ids.most_common(50000)]
print(ids_list)

sets = '/Users/chao/Desktop/code_dataset/demo/*'
files = sorted(list(glob(sets)))
print(files)
