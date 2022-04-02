
import networkx as nx
from infomap import Infomap

im = Infomap(silent=True)
im.add_node(1)
im.add_node(2)
im.add_link(1,2)
im.run()

out = im.get_modules()
print(out)


