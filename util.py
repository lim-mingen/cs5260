import os
import arxiv
import spacy
import numpy as np
from pyvis.network import Network
from itertools import combinations
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.cluster import AgglomerativeClustering

# load Pegasus summarizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model     = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# spaCy only for sentence‐splitting in concept‐map
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="sentence-transformers/allenai-specter")
embed_model = SentenceTransformer("sentence-transformers/allenai-specter")

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

def normalize_phrase(phrase: str) -> str:
    doc = nlp(phrase)
    lemmas = [tok.lemma_ for tok in doc if not tok.is_stop]
    return " ".join(lemmas)

def dedupe_by_substring(phrases):
    """
    phrases: list of (phrase, score) tuples sorted by score descending
    Returns a filtered list where no phrase is a substring of another.
    """
    filtered = []
    for ph, sc in phrases:
        # if any already-kept phrase contains this one, skip it
        if any(ph in kept for kept, _ in filtered):
            continue
        # if this phrase contains any already-kept shorter phrase, remove that shorter phrase
        filtered = [(k,s) for k,s in filtered if ph not in k]
        filtered.append((ph, sc))
    return filtered

def dedupe_by_embedding(phrases, threshold: float = 0.1):
    """
    phrases: list of (normalized_phrase, score) tuples
    threshold: cosine‐distance cutoff (lower = tighter clusters)
    """
    texts = [ph for ph, _ in phrases]
    # normalized embeddings, so cosine = dot product
    embs = embed_model.encode(texts, normalize_embeddings=True)

    # cluster by cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    ).fit(embs)

    clusters = {}
    for (ph, sc), lbl in zip(phrases, clustering.labels_):
        clusters.setdefault(lbl, []).append((ph, sc))

    # pick top scoring phrase per cluster
    result = [max(members, key=lambda x: x[1]) for members in clusters.values()]
    # sort by score
    return sorted(result, key=lambda x: x[1], reverse=True)

# +++ replace extract_entities with KeyBERT-based extraction +++
def extract_entities(text: str, top_n: int = 20):
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
    subphrases = dedupe_by_substring(raw_phrases)
    deduped = dedupe_by_embedding(subphrases)

    return [(ph, "KEYPHRASE") for ph, _ in deduped[:10]]

def build_concept_map(phrases, sim_threshold: float = 0.85) -> Network:
    """
    phrases: list of (text, label) tuples
    sim_threshold: cosine‐sim cutoff for adding an edge
    """
    net = Network(height="600px", width="100%")
    # 1) Add nodes
    id_map = {}
    texts  = [ph for ph, _ in phrases]
    for idx, (ph, lbl) in enumerate(phrases, start=1):
        id_map[ph] = idx
        net.add_node(idx, label=ph, title=lbl)
    # 2) Compute embeddings for all phrases
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # 3) Compare each pair
    for i, j in combinations(range(len(texts)), 2):
        sim = float(np.dot(embeddings[i], embeddings[j]))  # since normalized, dot=cosine
        print(f"sim({texts[i]}, {texts[j]}) = {sim:.3f}")
        if sim >= sim_threshold:
            net.add_edge(id_map[texts[i]], id_map[texts[j]], value=sim)

    net.set_options("""
    {
    "physics": {
        "solver": "repulsion",
        "repulsion": {
        "nodeDistance": 200,
        "springLength": 200,
        "damping": 0.5
        }
    }
    }
    """)
    return net
