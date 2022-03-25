format:
    <cited_paper_mag_id>␞<adjacent_citations_mag_ids>␞<citing_paper_mag_id>␞<cited_paper_arxiv_id>␞<adjacent_citations_arxiv_ids>␞<citing_paper_arxiv_id>␞<citation_context>
    (separated by a record separator (U+241E))

format <adjacent_citations_*_ids> if length == 0:
    empty

format <adjacent_citations_*_ids> if length == 1:
    <id>

format <adjacent_citations_*_ids> if length > 1:
    <id>␟<id>␟...
(separated by a unit separator (U+241F))

notes:
- adjacent_citations_mag_ids and adjacent_citations_arxiv_ids are, per line,
  always in the same order
- missing values (e.g. when a citing paper (which all have an arXiv ID) does
  not have a corresponding citing_paper_mag_id) are given as "None"
- to create context exports in different configurations (fewer/more sentences
  before/after the citing sentence etc.) use script code/extract_contexts.py
