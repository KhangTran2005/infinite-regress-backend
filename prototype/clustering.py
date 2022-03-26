#---- Must Install -----
# pys2, networkx, infomap, requests
# Query format: python -m clustering "[User query here]" depth (put 1 for testing)
#---- End Must Install ----

#TODO: Make the graph weighted -> use the similarity between any two abstracts/title/anything

import sys
import s2
import networkx as nx
from infomap import Infomap
import requests

def get_graph(s, G, depth=5, marked = {}):
  # Mark current node as visited
  marked[s] = True

  # Query API to get neigbours
  paper = s2.api.get_paper(s)
  neighbours = []
  for i in paper.citations + paper.references:
    neighbours.append(i.paperId)
  
  G[s] = neighbours

  if (depth == 0):
    return

  for v in neighbours:
    if v in marked: continue
    get_graph(v, G, depth - 1, marked)

def get_clustered_graph(G, depth_level=10):
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
    #Getting input
    arg = sys.argv
    user_in = arg[1]
    query = '+'.join(user_in.lower().split())
    depth = int(arg[2])
    if len(arg) == 4:
        depth_level = arg[3]
    else:
        depth_level = 10

    #Run the codes
    response = requests.get(f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=1')
    root = response.json()['data'][0]['paperId']

    graph_data = {}
    get_graph(root, graph_data, depth)
    G = nx.Graph(graph_data)
    print(get_clustered_graph(G, depth_level))
