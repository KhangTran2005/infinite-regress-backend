# sampling outline

- randomly select 100 papers from unarXive data set
- reduce reference item data base to only those entries where one of the
  sampled papers is citing or is being cited
- extract citation contexts for sampled reference items, restricted to sampled
  papers' full text (file `citation_contexts.csv`)
- extract citation contexts for sampled reference items, allowing use of full-
  text of all papers cotained in unarXive (file `citation_contexts_extended.csv`)


# sampling steps

### sampling
$ ls full/unarXive/papers/ > tmp/paper_list
$ cat tmp/paper_list | shuf | head -n 100 | sort > sample/paper_sample
$ cp full/unarXive/papers/refs.db .
$ python3 cull_refs_db.py
$ mkdir sample/papers
$ mv refs_sample.db papers/refs.db
$ while read p; do cp "full/unarXive/papers/$p" sample/papers/; done < sample/paper_sample

### extraction of contexts
$ python3 extract_contexts.py sample/papers/ -f citation_contexts.csv -u s -e 0 -o 0
$ python3 extract_contexts.py full/unarXive/papers/ -b sqlite:///sample/papers/refs.db -u s -e 0 -o 0 -f citation_contexts_extended.csv
