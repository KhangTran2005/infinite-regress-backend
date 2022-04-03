#Base Imports
import numpy as np
import pandas as pd
import pickle
import re
from infomap import Infomap
from functools import reduce
from multiprocessing.pool import ThreadPool
import networkx as nx
import sys
import os
import shutil

#API Calls
import arxiv
from arxiv.arxiv import SortCriterion

#Processing
from tika import parser
import bibtexparser

#NLP
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
t5model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5tokenizer = T5Tokenizer.from_pretrained("t5-base")

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#nltk.download('punkt')

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def escape_quotes(x):
    return x.replace('"','').replace("'","").replace(',',"").replace(':',"")

class Paper:
  def __init__(self, title, abstract, text=None):
    self.title = escape_quotes(title)
    self.abstract = abstract
    self.text = text
  
  def __eq__(self, other):
    return (self.title.lower() == other.title.lower()) and (self.abstract.lower() == other.abstract.lower())

  def __hash__(self):
    return hash(self.title + self.abstract)

  def __str__(self):
    return self.title

def __repr__(self):
    return str({'title': title,
            'abstract': abstract})

def summarize_corpus(corpus):
    txt = " ".join(corpus)
    tokens = t5tokenizer.encode(txt,return_tensors='pt',max_length=512,truncation=True)
    summary_ids = t5model.generate(tokens,min_length=60,max_length=180,length_penalty=4.0)
    try:
        return tokenizer.decode(summary_ids[0])
    except:
        return 'Unable to generate abstract'

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

  if citations is None:
      return
  def task(arg):
    return get_vertex(*arg)
  n = out_n if out_n != -1 else len(citations)

  pool = ThreadPool(n)
  results = pool.map(task, zip(citations[:n], [refimodel] * n, [naexmodel] * n, [nlp] * n))
  pool.close()
  pool.join()

  for paper, cites in results:
    if not paper:
        continue
    marked[paper.title] = paper, cites
    G.add_node(paper.title)
    w = get_weight_count(root, paper)
    G.add_edge(s_title, paper.title, weight=w)
    get_graph_recur_multi(paper.title, G, marked, refimodel, naexmodel, nlp, depth-1, out_n)
    

#Take 2 paper nodes and returns a weight
def get_weight_tfidf(p1, p2):
  corpus = [p1.title, p2.title, p1.abstract, p2.abstract]
  vect = TfidfVectorizer(min_df=1, stop_words='english')
  tfidf = vect.fit_transform(corpus)
  tfidf = pd.DataFrame(tfidf.toarray())
  t1, t2, a1, a2 = tfidf.iloc[0], tfidf.iloc[1], tfidf.iloc[2], tfidf.iloc[3]
  return 1/(1 + t1 @ t2 + a1 @ a2)

def get_weight_count(p1, p2):
  corpus = [p1.title, p2.title, p1.abstract, p2.abstract]
  vect = CountVectorizer(min_df=1, stop_words='english')
  count = vect.fit_transform(corpus)
  count = pd.DataFrame(count.toarray())
  t1, t2, a1, a2 = count.iloc[0], count.iloc[1], count.iloc[2], count.iloc[3]
  return float(f'{1/(1 + (t1/len(t1)) @ (t2/len(t2)) + (a1/len(a1)) @ (a2/len(a2))):.3f}')

#Given paper title, output citation list and paper node
def get_vertex(title, refimodel, naexmodel, nlp,pos=1):
  res = get_paper(title,pos=pos)
  if res != -1:
    title, abstract, text = res
  else:
    if pos < 10: 
        return get_vertex(title,refimodel,naexmodel,nlp,pos=pos+1)
    return Paper('not found', 'not found'), pd.Series()
  node = Paper(title, abstract)
  try:
    citations = get_citations(text, refimodel, naexmodel, nlp)
  except:
    citations = text
  if citations is None:
      return get_vertex(title,refimodel,naexmodel,nlp,pos=pos+1)
  return node, citations

#Given raw text get citation list
def get_citations(text, refimodel, naexmodel, nlp):
  _, _, _, ref, _ = segment_doc(text)

  #Processing the reference chunks and finding valid chunks
  ref_list = get_ref_list(ref)
  refs = clean_ref(ref_list)  
  refs[refs.str.match('^\s*?\[\d+?\]')] = refs[refs.str.match('^\s*?\[\d+?\]')].str.rsplit(r']').apply(lambda x: ']'.join(x[1:]).strip())
  refs[refs.str.match('\s*?\d+?\.\s+?')] = refs[refs.str.match('\s*?\d+?\.\s+?')].str.split(r'\. ').apply(lambda x: '. '.join(x[1:])).str.strip()
  refs[refs.str.match('\s*?\d+?\s+?')] = refs[refs.str.match('\s*?\d+?\s+?')].str.split(r'^\s*?\d+?\s+?').apply(lambda x: ' '.join(x[1:])).str.strip()
  X_refs = get_X(refs, nlp)
  y_pred = refimodel.predict(X_refs) == 0
  #print(refs)
  #print('---cum---')
  #print(refs[y_pred])
  refs = refs[y_pred]

  #Identifying the paper names in each reference chunk
  sent = get_sent(refs)
  X_sent = get_X(sent, nlp)
  name_list = sent[naexmodel.predict(X_sent) == 1]
  name_list = name_list.str.split('[“”]').apply(lambda x: ''.join(x)).str.strip()
  name_list = name_list.str.split(', ').apply(lambda x: ', '.join(x[1:]) if(len(x[0]) < 20) else ', '.join(x)).str.strip()
  name_list = name_list[[len([word for word in word_tokenize(i) if word.isalnum()]) > 4 for i in name_list]]
  return name_list.reset_index()[0]

def get_sent(refs):
  return pd.Series(reduce(lambda x, y: x + y, refs.apply(lambda x: x.split('. '))))

#Given a title, get a paper's title, abstract, and text from an API
def get_paper(title, walker = 1,pos=1):
  try:
    res = arxiv.Search(query=f'{title}', max_results=walker).results()
    while pos:
        paper = next(res)
        pos -= 1
  except Exception as e:
    return -1
  count = 1
  while paper.title.lower() != title.lower() and count < walker:
    paper = next(res)
    count += 1
  try:
    paper.download_pdf(filename=f'../paper-cache/{title}.pdf')
    text = parser.from_file(f'../paper-cache/{title}.pdf')
  except:
    text = {'content': ''}
  return paper.title, paper.summary, text['content']

#Segment pdf parse into header, abstract, body, ref, and appendix (if any)
def segment_doc(doc):
  split = re.split(r'\n *?(?:Abstract|ABSTRACT)\s*?', doc)
  if len(split) > 1:
    header, rest = split[0], '\n'.join(split[1:])
  else:
    header = ''
    rest = doc
  split_2 = re.split(r'\n(?:[\w\.]*?|\d)\s*?(?:Introduction|INTRODUCTION)\s*?\n', rest)
  if len(split_2) > 1:
    abstract, rest = split_2[0], split_2[1]
  else:
    split_2 = re.split(r'^([\w\W\S_]*)\n{3,}?', rest.strip())
    abstract, rest = split_2[0], split_2[1]
  split_3 = re.split('\n\d?\.?\s?(?:Bibliographical )?(?:References|REFERENCES|LITERATURE CITED|REFERENCE)\s*?\n', rest)
  if len(split_3) > 1:
    body, rest = split_3[0], split_3[1]
  else:
    body = ''
  try:
    references, appendix = tuple(re.split(r'\n\n(?:A\.?\s|Appendices|Appendix).*\n*?', rest))
  except:
    references = re.split(r'\n{3,}', rest)[0]
    appendix = ''
  return header.strip(), abstract.strip(), body.strip(), references.strip(), appendix.strip()

#Functions to get further segment the references section
def reformat_citations(original):
  ref = ' '.join(re.split(r'\n', original))
  ref = ''.join(re.split(r'- ', ref))
  return ref
def get_ref_list(ref):
  return np.array([reformat_citations(i) for i in ref.split('\n\n') if i[:5] != 'https'])

#Cleaning out entries that are obviously not citations as well as fixing up few errors
def clean_ref(ref_list):
  non_empty = ref_list[ref_list != '']
  refs = pd.Series(non_empty)
  dash = refs[refs.str.endswith('-')]
  merged = []
  for i in dash.index:
    next = refs[i + 1]
    merged.append(refs[i][:-1] + next)
  refs = refs.drop(index=dash.index.append(dash.index + 1)).append(pd.Series(merged)).reset_index()[0]
  merged = []
  idx = refs[refs.str.startswith('and')].index
  for i in idx:
    prev = refs[i - 1]
    merged.append(refs[i - 1] + ' ' + refs[i])
  #Remove non english
  non_eng = refs[refs.str.match(r'.*[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u3131-\uD79D]')].index
  refs = refs.drop(non_eng).reset_index()[0]
  refs = refs[~refs.str.match('[a-z]\.')].reset_index()[0]
  return refs

#Function to return the features
def get_features(doc):
  #doc = nlp(text)
  person = 0
  others = 0
  for i in doc.ents:
    if i.label_ == 'PERSON':
      person += 1
    else:
      others += 1
  propn = 0
  noun = 0
  adp = 0
  adj = 0
  punc = 0
  count = 0
  sym = 0
  for token in doc:
    count += 1
    if token.pos_ == 'PROPN':
      propn += 1
    elif token.pos_ == 'NOUN':
      noun += 1
    elif token.pos_ == 'PUNCT':
      punc += 1
    elif token.pos_ == 'ADP':
      adp += 1
    elif token.pos_ == 'ADJ':
      adj += 1
    elif token.pos_ == 'SYM':
      sym += 1
  return {
      'person': person,
      'others': others,
      'propn': propn,
      'noun': noun,
      'adp': adp,
      'adj': adj,
      'punc': punc,
      'count': count,
      'sym': sym
  }

#Getting the characteristic matrix (matrix where each column is a feature)
def get_X(ref4s, nlp):
  pipe = nlp.pipe(ref4s)
  features = get_features(next(pipe))
  X = [list(features.values())]
  feature_names = list(features.keys())
  for doc in pipe:
    X.append(list(get_features(doc).values()))
  return pd.DataFrame(np.array(X), columns=feature_names)

#Given a networkx graph, return a clustered representation
def get_clustered_graph(G, depth_level=5):
    im = Infomap(silent=True)
    mapping = im.add_networkx_graph(G)
    im.run()
    clustering = im.get_modules()
    id = mapping.values()
    clust = clustering.values()
    graph_data = {}
    graph_data["nodes"] = [{"id": escape_quotes(a), "group": b} for a, b in zip(id, clust)]
    graph_data["links"] = [{"source": escape_quotes(a), "target": escape_quotes(b), "value": 1} for a, b in G.edges]
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
    if os.path.exists('paper-cache'):
        shutil.rmtree('paper-cache')
    lpp = os.path.abspath('label-propagation')
    stcp = os.path.abspath('stc')
    lp = pickle.load(open(lpp, 'rb'))
    stc = pickle.load(open(stcp, 'rb'))
    try:
        os.mkdir('paper-cache')
    except:
        shutil.rmtree('paper-cache')
        os.mkdir('paper-cache')

    marked, G = get_graph_multi(user_in, lp, stc, depth=depth, out_n=out_n)
    graph_data = get_clustered_graph(G, depth_level)
    paper_data = {}
    id2abstract = dict()
    for v in marked:
        paper, _ = marked[v]
        paper_data[v] = escape_quotes(paper.abstract)
        id2abstract[paper.id] = paper.abstract
    clusters = [[] for _ in range(len(graph_data['nodes']))]
    for node in graph_data['nodes']:
        clusters[node['group']].append(id2abstract[node['id']])
    summaries = []
    for i,c in enumerate(clusters):
        if len(c) > 0:
            summaries.append({
                'cluster': i,
                'summary': summarize_corpus(c)
                })
    #print(graph_data)
    print({
      'graph_data': graph_data,
      'paper_data': paper_data,
      'cluster_data': summaries
    })
    
    shutil.rmtree('paper-cache')

