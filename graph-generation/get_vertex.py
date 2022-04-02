#Base Imports
import numpy as np
import pandas as pd
import re
from functools import reduce
from paper import Paper

#API Calls
import arxiv
from arxiv.arxiv import SortCriterion

#Processing
from tika import parser

#NLP
import spacy
import nltk
from nltk.tokenize import word_tokenize
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')



#Given paper title, output citation list and paper node
def get_vertex(title, refimodel, naexmodel, nlp):
  res = get_paper(title)
  if res != -1:
    title, abstract, text = res
  else:
    return Paper('not found', 'not found'), pd.Series()
  node = Paper(title, abstract)
  try:
    citations = get_citations(text, refimodel, naexmodel, nlp)
  except:
    citations = text
  return node, citations

#Given raw text get citation list
def get_citations(text, refimodel, naexmodel, nlp):
  _, _, _, ref, _ = segment_doc(text)

  #Processing the reference chunks and finding valid chunks
  ref_list = get_ref_list(ref)
  refs = clean_ref(ref_list)  
  X_refs = get_X(refs, nlp)
  refs = refs[refimodel.predict(X_refs) == 1]
  refs[refs.str.match('^\s*?\[\d+?\]')] = refs[refs.str.match('^\s*?\[\d+?\]')].str.rsplit(r']').apply(lambda x: ']'.join(x).strip())
  refs[refs.str.match('\s*?\d+?\.\s+?')] = refs[refs.str.match('\s*?\d+?\.\s+?')].str.split(r'\. ').apply(lambda x: '. '.join(x)).str.strip()
  refs[refs.str.match('\s*?\d+?\s+?')] = refs[refs.str.match('\s*?\d+?\s+?')].str.split(r'^\s*?\d+?\s+?').apply(lambda x: ' '.join(x)).str.strip()

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
def get_paper(title, walker = 1):
  try:
    res = arxiv.Search(query=f'{title}', max_results=walker).results()
    paper = next(res)
  except Exception as e:
    print(e)
    return -1

  count = 1
  while paper.title.lower() != title.lower() and count < walker:
    paper = next(res)
    print(count)
    count += 1
  try:
    paper.download_pdf(filename=f'{title}.pdf')
    text = parser.from_file(f'{title}.pdf')
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
def get_features(text, nlp):
  doc = nlp(text)
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
  features = get_features(ref4s[0], nlp)
  X = [list(features.values())]
  feature_names = list(features.keys())
  for i in ref4s[1:]:
    X.append(list(get_features(i, nlp).values()))
  return pd.DataFrame(np.array(X), columns=feature_names)