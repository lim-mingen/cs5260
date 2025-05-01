import os
import arxiv
import spacy
import numpy as np
from pyvis.network import Network
from itertools import combinations
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.cluster import AgglomerativeClustering
from semanticscholar import SemanticScholar
from habanero import Crossref
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

sch = SemanticScholar(timeout=30)
cr = Crossref(mailto="limmingen95@gmail.com")

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="sentence-transformers/allenai-specter")
embed_model = SentenceTransformer("sentence-transformers/allenai-specter")

load_dotenv(dotenv_path="config/.env")
client = OpenAI(
    api_key=os.getenv("API_KEY"),         
    base_url="https://api.deepinfra.com/v1/openai"
)

def fetch_arxiv(query, max_results=5):
    # Scrape papers from arXiv
    search = arxiv.Search(query=query, max_results=max_results)
    return [{
        "entry_id": r.entry_id.split("/")[-1],
        "title":    r.title,
        "abstract": r.summary
    } for r in search.results()]

def fetch_semantic_scholar(query, max_results=5): 
    # Scrape papers from Semantic Scholar
    # Note: Semantic Scholar API does not return abstracts for all papers
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
    # Scrape papers from CrossRef
    # Note: CrossRef API does not return abstracts for all papers
    items = cr.works(query=query, limit=max_results)["message"]["items"]
    return [{
        "entry_id": itm.get("DOI", str(i)),
        "title":    itm.get("title", [""])[0],
        "abstract": itm.get("abstract", "")
    } for i, itm in enumerate(items)]

def summarize_abstract_spacy(text: str, num_sentences: int = 3) -> str:
    # Summarize abstracts via spaCy's en_core_web_sm
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

def dedupe_by_substring(phrases):
    # Remove phrases that are substrings of others. Used in keyphrase extraction.
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
    # Remove phrase that are too similar to others. Used in keyphrase extraction.
    texts = [ph for ph, _ in phrases]
    embs = embed_model.encode(texts, normalize_embeddings=True)

    # Cluster by cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    ).fit(embs)

    clusters = {}
    for (ph, sc), lbl in zip(phrases, clustering.labels_):
        clusters.setdefault(lbl, []).append((ph, sc))

    # Pick top scoring phrase per cluster
    result = [max(members, key=lambda x: x[1]) for members in clusters.values()]
    return sorted(result, key=lambda x: x[1], reverse=True)

def extract_entities(text: str, top_n: int = 20):
    # Use Specter model via KeyBERT to extract keyphrases
    raw_phrases = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n
    )
    # Remove duplicates and too-similar phrases
    subphrases = dedupe_by_substring(raw_phrases)
    deduped = dedupe_by_embedding(subphrases)

    return [(ph, "KEYPHRASE") for ph, _ in deduped[:10]]

def summarize_abstracts_llm(
    abstracts: List[str],
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    # Cross-paper summary using Qwen model
    prompt = (
        f"These are the abstracts of {len(abstracts)} papers. "
        "Produce a cross-paper summary that summarizes all the key points across each paper. Keep it to 5-6 sentences.\n\n"
    )
    for i, abs_text in enumerate(abstracts, start=1):
        prompt += f"Paper {i} abstract:\n{abs_text}\n\n"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful academic research assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def build_global_concept_map(papers):
    # Global concept map of scraped papers
    
    # Map node to title fpr tooltip
    phrase_to_titles = {}
    for p in papers:
        ents = extract_entities(p["abstract"])
        phrases = {e for e,_ in ents}
        for ph in phrases:
            phrase_to_titles.setdefault(ph, []).append(p["title"])

    freq = Counter()
    for ph, titles in phrase_to_titles.items():
        freq[ph] = len(titles)

    net = Network(height="600px", width="100%")
    id_map = {ph: idx for idx, ph in enumerate(freq, start=1)}

    for ph, count in freq.items():
        titles = phrase_to_titles.get(ph, [])
        tooltip = "<br>".join(titles)
        net.add_node(
            id_map[ph],
            label=ph,
            title=tooltip,
            size=10 + 2 * count
        )

    cooc = Counter()

    per_paper_sets = []
    for p in papers:
        ents = extract_entities(p["abstract"])
        per_paper_sets.append({e for e,_ in ents})

    for phrases in per_paper_sets:
        for a, b in combinations(sorted(phrases), 2):
            cooc[(a, b)] += 1

    for (a, b), c in cooc.items():
        net.add_edge(id_map[a], id_map[b], value=c)

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
    # Individual concept map of each paper. Threshold set to 0.85.
    net = Network(height="600px", width="100%")

    id_map = {}
    texts  = [ph for ph, _ in phrases]
    for idx, (ph, lbl) in enumerate(phrases, start=1):
        id_map[ph] = idx
        net.add_node(idx, label=ph, title=lbl)

    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

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
