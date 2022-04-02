#Installs
# !pip install arxiv
# !pip install tika
# !pip install infomap
#Installs

#Imports

#Custom Imports
from paper import Paper

#Base Imports
import numpy as np
import pandas as pd
import pickle
from get_vertex import get_vertex
from multiprocessing.pool import ThreadPool

import spacy
from infomap import Infomap
from weights import *
import networkx as nx
import sys
import arxiv

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#End Imports

#Constructs the graph given a starting title
def get_graph_multi(s_title, refimodel, naexmodel, depth=5, out_n = 5):
  #Get paper node and citations
  nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "attribute_ruler", "lemmatizer"])
  marked = {s_title: get_vertex(s_title, refimodel, naexmodel, nlp)}
  G = nx.Graph()
  G.add_node(s_title)

  get_graph_recur_multi(s_title, G, marked, refimodel, naexmodel, nlp, depth, out_n)

  return marked, G

def get_graph_recur_multi(s_title, G, marked, refimodel, naexmodel, nlp, depth=5, out_n=5):
  if depth == 0:
    return

  root, citations = marked[s_title]

  #Perhaps we can add another function here to add all leaf nodes into the thing

  def task(arg):
    return get_vertex(*arg)

  n = out_n if out_n != -1 else len(citations)
  pool = ThreadPool(n)
  results = pool.map(task, zip(citations[:n], [refimodel] * n, [naexmodel] * n, [nlp] * n))
  pool.close()
  pool.join()
  for paper, cites in results:
    marked[paper.title] = paper, cites
    G.add_node(paper.title)
    w = get_weight_count(root, paper)
    G.add_edge(s_title, paper.title, weight=w)
    get_graph_recur_multi(paper.title, G, marked, refimodel, naexmodel, nlp, depth-1, out_n)

#Given a networkx graph, return a clustered representation
def get_clustered_graph(G, depth_level=5):
    im = Infomap(silent=True)
    mapping = im.add_networkx_graph(G)
    im.run()
    clustering = im.get_modules(depth_level=depth_level)
    id = mapping.values()
    clust = clustering.values()
    graph_data = {}
    graph_data["nodes"] = [{"id": a, "group": b} for a, b in zip(id, clust)]
    graph_data["links"] = [{"source": a, "target": b, "value": 1} for a, b in G.edges]
    return graph_data

if __name__ == '__main__':
    arg = sys.argv
    user_in = arg[1]
    depth = int(arg[2])
    out_n = int(arg[3])
    if len(arg) == 5:
        depth_level = arg[4]
    else:
        depth_level = 5
    
    lp = pickle.load(open('../ref-iden-models/label-propagation', 'rb'))
    stc = pickle.load(open('../name-extrac/stc', 'rb'))
    marked, G = get_graph_multi(user_in, lp, stc, depth=depth, out_n=out_n)
    print(get_clustered_graph(G, depth_level))
    print('------------Graph Info------------')
    print(marked)
