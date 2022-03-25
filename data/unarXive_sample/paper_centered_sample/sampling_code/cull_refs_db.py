import sqlite3

sample_ids = []
with open('sample/paper_sample') as f:
    for line in f:
        aid = line.strip().replace('.txt', '')
        sample_ids.append(aid)

conn_full = sqlite3.connect('refs.db')
cf = conn_full.cursor()
sample_bibitem_rows = []
uuids_in = []
for i, sid in enumerate(sample_ids):
    print('{}/{}'.format(i, len(sample_ids)))
    out_bibitems = cf.execute(
        'select * from bibitem where citing_arxiv_id=?',
        (sid,)
        ).fetchall()
    in_bibitems = cf.execute(
        'select * from bibitem where cited_arxiv_id=?',
        (sid,)
        ).fetchall()
    for row in out_bibitems + in_bibitems:
        uuid = row[0]
        if uuid in uuids_in:
            continue
        sample_bibitem_rows.append(row)
        uuids_in.append(uuid)

conn_sample = sqlite3.connect('refs_sample.db')
cs = conn_sample.cursor()
cs.execute('''CREATE TABLE bibitem (
        uuid VARCHAR(36) NOT NULL,
        citing_mag_id VARCHAR(36),
        cited_mag_id VARCHAR(36),
        citing_arxiv_id VARCHAR(36),
        cited_arxiv_id VARCHAR(36),
        bibitem_string TEXT,
        PRIMARY KEY (uuid)
);''')
cs.executemany(
    'insert into bibitem values (?,?,?,?,?,?)',
    sample_bibitem_rows
    )
conn_sample.commit()
conn_full.close()
conn_sample.close()
