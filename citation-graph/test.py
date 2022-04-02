import os
import pickle
import shutil

shutil.rmtree('dick')
p = os.path.abspath('citation-graph/label-propagation')
pickle.load(open(p, 'rb'))