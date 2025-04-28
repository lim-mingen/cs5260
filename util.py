# util.py
import arxiv
from transformers import pipeline
import spacy
from pyvis.network import Network

# 1. Data Collection: fetch papers from arXiv
def fetch_papers(query, max_results=5):
    """
    Search arXiv for papers matching `query` and return a list of dicts
    containing title, abstract, and URL.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id,
        })
    return papers

# 2. Summarization: using a Pegasus model
summarizer = pipeline("summarization", model="google/pegasus-xsum")

def summarize_text(text):
    """
    Summarize input text into a concise paragraph.
    """
    out = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return out[0]["summary_text"]

# 3. Visualization: extract entities via spaCy and build a concept map with PyVis
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Run NER on text and return list of (entity, label).
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def build_concept_map(entities):
    """
    Create a simple directed graph linking entities sequentially.
    Returns a PyVis Network object.
    """
    net = Network(height="600px", width="100%", directed=True)
    seen = set()
    for text, label in entities:
        if text not in seen:
            net.add_node(text, label=text, title=label)
            seen.add(text)
    for i in range(len(entities) - 1):
        src = entities[i][0]
        dst = entities[i+1][0]
        net.add_edge(src, dst)
    return net