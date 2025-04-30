from semanticscholar import SemanticScholar
import time
from itertools import islice


sch = SemanticScholar(timeout=30)

def fetch_semantic_scholar(query, max_results=5):
    paginated = sch.search_paper(query, fields=['title'], limit=max_results) 
    papers = []
    for paper in islice(paginated, max_results):
        abstract = paper.abstract or ""
        if not abstract.strip():
            continue
        papers.append({
            "entry_id": paper.paperId,
            "title":    paper.title,
            "abstract": abstract
        })
    return papers

query = "adversarial machine learning"
papers = fetch_semantic_scholar(query, max_results=99)
for paper in papers:
    print(paper["title"])
    print(paper["abstract"])
    print()