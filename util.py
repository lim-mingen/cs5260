# utils.py

import os
import arxiv
import requests
import PyPDF2
import spacy
from pyvis.network import Network
from itertools import combinations
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# +++ new imports +++
from keybert import KeyBERT

# ensure folders
PDF_DIR  = "pdf"
HTML_DIR = "output"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# load Pegasus summarizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model     = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# spaCy only for sentence‐splitting in concept‐map
nlp = spacy.load("en_core_web_sm")

# +++ Initialize KeyBERT with SciBERT embeddings +++
# this will download the SciBERT model under the hood
kw_model = KeyBERT(model="allenai/scibert_scivocab_uncased")

def fetch_papers(query: str, max_results: int = 5):
    search = arxiv.Search(query=query, max_results=max_results)
    papers = []
    for result in search.results():
        papers.append({
            "entry_id": result.entry_id.split("/")[-1],
            "title":    result.title,
            "abstract": result.summary,
            "pdf_url":  result.pdf_url.replace("/abs/", "/pdf/") + ".pdf"
        })
    return papers

def summarize_abstract_spacy(text: str, num_sentences: int = 3) -> str:
    # (same as before) extractive summarizer via spaCy
    doc = nlp(text)
    freqs = {}
    for tok in doc:
        if tok.is_stop or tok.is_punct or not tok.is_alpha:
            continue
        w = tok.text.lower()
        freqs[w] = freqs.get(w, 0) + 1
    if not freqs: return ""
    maxf = max(freqs.values())
    for w in freqs: freqs[w] /= maxf

    sent_scores = {
        sent: sum(freqs.get(tok.text.lower(),0) for tok in sent if tok.is_alpha)
        for sent in doc.sents
    }
    # pick top sentences
    best = sorted(sent_scores, key=sent_scores.get, reverse=True)[:num_sentences]
    best_sorted = sorted(best, key=lambda s: list(doc.sents).index(s))
    return " ".join(s.text.strip() for s in best_sorted)

# +++ replace extract_entities with KeyBERT-based extraction +++
def extract_entities(text: str, top_n: int = 10):
    """
    Use SciBERT via KeyBERT to get top_n keyphrases, 
    then drop any phrase containing a VERB token.
    """
    raw_phrases = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n
    )
    filtered = []
    for phrase, score in raw_phrases:
        doc = nlp(phrase)
        # skip if *any* token in the phrase is a verb
        if any(tok.pos_ == "VERB" for tok in doc):
            continue
        filtered.append((phrase, "KEYPHRASE"))
    return filtered

def build_concept_map(phrases, min_cooccurrence=1):
    """
    phrases: list of (phrase_text, label) tuples
    min_cooccurrence: only draw edges for pairs seen together at least this many times
    """
    # 1) Build a mapping from phrase→node_id
    net    = Network(height="600px", width="100%")
    id_map = {}
    for i, (ph, lbl) in enumerate(phrases, start=1):
        id_map[ph] = i
        net.add_node(i, label=ph, title=lbl)

    # 2) Parse the combined text to get sentence boundaries
    text = " ".join(ph for ph, _ in phrases)
    doc  = nlp(text)

    # 3) Count co-occurrences in each sentence
    cooc = {}
    for sent in doc.sents:
        # find which phrases appear in this sentence
        present = [ph for ph, _ in phrases if ph in sent.text]
        for a, b in combinations(present, 2):
            pair = tuple(sorted((a,b)))
            cooc[pair] = cooc.get(pair, 0) + 1

    # 4) Add edges for pairs above your threshold
    for (a, b), count in cooc.items():
        if count >= min_cooccurrence:
            net.add_edge(id_map[a], id_map[b], value=count)

    # 5) (Optional) Tune physics so thicker edges pull nodes closer
    net.set_options("""
    {
      "physics": {"forceAtlas2Based": {"gravitationalConstant": -50}},
      "edges": { "smooth": false }
    }
    """)
    return net

def save_graph_html(net: Network, name: str) -> str:
    path = os.path.join(HTML_DIR, f"{name}.html")
    net.write_html(path)
    return path
