# Academic Paper Summarizer & Concept-Map Explorer

A lightweight Gradio dashboard to help AI/ML researchers quickly find, summarize, and visualize the conceptual landscape of academic papers.  

- 🔍 **Search** ArXiv by keyword  
- ✂️ **Per-paper summary** (2 – 3 sentences) via spaCy extractive summarization  
- 🤖 **Cross-paper summary** (5 – 6 sentences) driven by **Qwen/Qwen2.5-Coder-32B-Instruct**  
- 🌐 **Global concept map** (all papers) and 📝 **per-paper concept maps** via KeyBERT + Sentence-Transformer embeddings + PyVis  

---

## 📂 Repository Layout

- util.py: contains core functions to summarize, extract and build concept map
- main.py: contains Gradio UI functions
- config
   - .env: holds API_KEY to access DeepInfra OpenAI
- requirements.txt
- README.md

---

## ⚙️ Installation

1. **Clone** the repo and enter its folder  
   ```bash
      git clone https://github.com/lim-mingen/cs5260.git
      cd cs5260
   
2. Create a virtual environment and install
   ```bash
      pip install -r requirements.txt

3. Add your DeepInfra API key in confif/.env

4. Run the app
   ```bash
      python main.py

5. Open the URL printed in your terminal to start exploring

## 🛠️ Features & Methodology
1. Data Collection
Source: arXiv via the arxiv Python library

(Disabled): Semantic Scholar & CrossRef wrappers included, but commented out since many entries lack abstracts

2. Per-Paper Summarization
Model: spaCy en_core_web_sm

Steps: 
a. Tokenize & filter stop-words/punctuation
b. Score sentences by term-frequency
c. Select top 2–3 sentences

3. Keyphrase Extraction & Concept Maps
Keyphrases: extracted with KeyBERT over Specter embeddings
Deduplication: 
a. Substring-based filtering
b. Agglomerative clustering on normalized embeddings (cosine threshold = 0.1)

4. Cross-Paper Summary
Model: Qwen/Qwen2.5-Coder-32B-Instruct via DeepInfra’s OpenAI-compatible endpoint
Prompt:
"These are the abstracts of N papers. Produce a cross-paper summary that summarizes all the key points across each paper. Keep it to 5-6 sentences."
Output: A cohesive 5–6 sentence narrative covering all fetched abstracts

5. Global Concept Network
Nodes: every extracted keyphrase, sized by how many papers mention it
Edges: co-occurrence within individual papers
Hover tooltip: lists paper titles containing that phrase

Graphs (PyVis):
Nodes: top 10 keyphrases per paper
Edges: connect if cosine similarity ≥ 0.85
Layout: force-directed repulsion (nodeDistance, springLength, damping)

## 🔬 Experiments & Dead-Ends
Semantic Scholar & CrossRef
• Added fetch_semantic_scholar and fetch_crossref with semanticscholar/habanero clients
• Outcome: most results lacked abstracts or relevance → disabled

Full-Text PDF Extraction
• Downloaded PDFs + PyPDF2 → NER/summarization on full text
• Outcome: noisy extractions from captions, tables, references → reverted to abstracts only

Domain-Specific NER
• Tried SciSpaCy (biomedical) and SciERC transformers
• Outcome: labels too niche or model download failures → stuck with spaCy general NER

Keyphrase Approaches
• RAKE, TextRank, KeyBERT with SciBERT embeddings
• Outcome: required heavy verb/digit filtering and complex clustering → current pipeline balances precision & simplicity

Cross-Paper Summarizers
• Pegasus-XSum (single sentence) → too terse
• BART-CNN hierarchical summarization → 3–5 sentences but lacked coherence
• Solution: LLM prompt via Qwen2.5-Coder-32B-Instruct gave the best narrative

Concept-Map Connectivity
• Co-occurrence in sentences → per-paper clusters only
• Embedding similarity edges → performance/hair-ball issues
• Final: per-paper maps by embedding similarity (threshold 0.85) + a single global map by co-occurrence
