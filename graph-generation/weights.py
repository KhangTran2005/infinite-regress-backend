#Take 2 paper nodes and returns a weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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