import os
import arxiv
import spacy
import numpy as np
from pyvis.network import Network
from itertools import combinations
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.cluster import AgglomerativeClustering
from semanticscholar import SemanticScholar
from habanero import Crossref
from collections import Counter
from sklearn.cluster import KMeans
import arxiv
from typing import List

sch = SemanticScholar(timeout=30)
cr = Crossref(mailto="limmingen95@gmail.com")

# load Pegasus summarizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model     = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model     = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# spaCy only for sentence‐splitting in concept‐map
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="sentence-transformers/allenai-specter")
embed_model = SentenceTransformer("sentence-transformers/allenai-specter")

def fetch_arxiv(query, max_results=5):
    search = arxiv.Search(query=query, max_results=max_results)
    return [{
        "entry_id": r.entry_id.split("/")[-1],
        "title":    r.title,
        "abstract": r.summary
    } for r in search.results()]

def fetch_semantic_scholar(query, max_results=5): 
    paginated = sch.search_paper(query, fields=['title'], limit=max_results) 
    first_page = paginated.items
    papers = []
    for paper in first_page:
        papers.append({
            "entry_id": paper.paperId,
            "title":    paper.title,
            "abstract": paper.abstract or ""
        })
    return papers

def fetch_crossref(query, max_results=5):
    items = cr.works(query=query, limit=max_results)["message"]["items"]
    return [{
        "entry_id": itm.get("DOI", str(i)),
        "title":    itm.get("title", [""])[0],
        "abstract": itm.get("abstract", "")
    } for i, itm in enumerate(items)]

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

def cluster_abstracts(abstracts, max_clusters=5):
    """
    Cluster the list of abstracts into up to max_clusters using KMeans on Sci-paper embeddings.
    Returns a dict: {cluster_label: [abstract1, abstract2, ...], ...}
    """
    n = len(abstracts)
    k = min(max_clusters, n)
    embs = embed_model.encode(abstracts, normalize_embeddings=True)
    km   = KMeans(n_clusters=k, random_state=0).fit(embs)
    clusters = {}
    for idx, label in enumerate(km.labels_):
        clusters.setdefault(label, []).append(abstracts[idx])
    return clusters

def summarize_clusters(clusters):
    """
    Take a dict of clusters→[abstracts], summarize each cluster as one text block.
    Returns: {cluster_label: summary_text, ...}
    """
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    tok = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    m   = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    summaries = {}
    for label, abs_list in clusters.items():
        text = " ".join(abs_list)
        inputs = tok(text, truncation=True, padding="longest", return_tensors="pt")
        ids = m.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
        summaries[label] = tok.decode(ids[0], skip_special_tokens=True)
    return summaries

def build_narrative(cluster_summaries):
    """
    Given cluster summaries dict, write an overarching narrative as markdown.
    """
    k = len(cluster_summaries)
    parts = [f"We identified **{k}** major themes across the retrieved papers:\n"]
    for label, summ in cluster_summaries.items():
        parts.append(f"- **Theme {label+1}**: {summ}")
    return "\n\n".join(parts)

def summarize_text_multi(text: str, max_length: int = 200, min_length: int = 80) -> str:
    inputs = bart_tokenizer(
        text,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        max_length=1024,      # BART’s max input
    )
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length,
        min_length=min_length,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_abstracts_batch(
        abstracts: List[str],
        max_length: int = 200,
        chunk_size: int = 3
    ) -> str:
    """
    Hierarchical summarization of a list of abstracts:
      1) Break into chunks of `chunk_size`
      2) Summarize each chunk (max_length tokens)
      3) Concatenate chunk-summaries and summarize again
    """
    # 1) Summarize each chunk
    intermediate_summaries = []
    for i in range(0, len(abstracts), chunk_size):
        chunk = abstracts[i : i + chunk_size]
        text  = "\n\n".join(chunk)
        # use your existing PEGASUS summarizer
        sum_chunk = summarize_text_multi(text, max_length=max_length)
        intermediate_summaries.append(sum_chunk)

    # 2) Combine intermediate summaries
    combined = "\n\n".join(intermediate_summaries)

    # 3) Final pass
    final_summary = summarize_text_multi(combined, max_length=max_length)
    return final_summary

def build_global_concept_map(papers):
    """
    Aggregate keyphrases from all papers, size nodes by frequency,
    connect co-occurring phrases, and show paper titles on hover.
    """
    # 1) Map each phrase to the set of paper titles where it appears
    phrase_to_titles = {}
    for p in papers:
        ents = extract_entities(p["abstract"])
        phrases = {e for e,_ in ents}
        for ph in phrases:
            phrase_to_titles.setdefault(ph, []).append(p["title"])

    # 2) Count overall frequencies
    all_phrases = [ph for titles in phrase_to_titles.values() for ph in titles]
    # Actually we want frequencies of phrases, so:
    freq = Counter()
    for ph, titles in phrase_to_titles.items():
        freq[ph] = len(titles)

    # 3) Build the network
    net = Network(height="600px", width="100%")
    id_map = {ph: idx for idx, ph in enumerate(freq, start=1)}

    # Add nodes with tooltip = list of titles
    for ph, count in freq.items():
        titles = phrase_to_titles.get(ph, [])
        # join titles with HTML line breaks
        tooltip = "<br>".join(titles)
        net.add_node(
            id_map[ph],
            label=ph,
            title=tooltip,
            size=10 + 2 * count
        )

    # 4) Co-occurrence edges (per paper)
    cooc = Counter()
    for titles in phrase_to_titles.values():
        # we actually need to rebuild per-paper phrase sets:
        pass  # we'll rebuild below

    # Better: rebuild per-paper sets to count co-occurrence
    per_paper_sets = []
    for p in papers:
        ents = extract_entities(p["abstract"])
        per_paper_sets.append({e for e,_ in ents})

    for phrases in per_paper_sets:
        for a, b in combinations(sorted(phrases), 2):
            cooc[(a, b)] += 1

    for (a, b), c in cooc.items():
        net.add_edge(id_map[a], id_map[b], value=c)

    # 5) Tweak physics for more space
    net.set_options("""
    {
      "physics": {
        "solver": "repulsion",
        "repulsion": {
          "nodeDistance": 250,
          "springLength": 200,
          "damping": 0.5
        }
      }
    }
    """)
    return net

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
