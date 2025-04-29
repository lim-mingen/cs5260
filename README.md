# Academic Paper Summarization & Visualization Tool

## Overview

This project provides an AI-powered system to automatically read academic research papers, extract and summarize their key points, and visualize the underlying concepts as interactive diagrams or concept maps. Users can further annotate and comment on the generated visualizations via a web-based interface.

## Methodology

The workflow is divided into four main stages:

1. **Data Collection**
   - Fetch research papers programmatically from arXiv via the `arxiv` Python API.
   - (Optional) Extend to other sources like Semantic Scholar or custom web-scraping pipelines.

2. **Summarization**
   - Use a pre-trained transformer model (Google PEGASUS `pegasus-xsum`) to generate concise summaries of paper abstracts.
   - Fine-tune or swap to other models (e.g., BLOOM) for domain-specific performance.

3. **Visualization**
   - Run Named Entity Recognition (NER) with spaCy (`en_core_sci_md`) to identify key terms and concepts.
   - Construct an interactive concept map using PyVis, linking extracted entities to reveal relationships.

4. **Annotation & Dashboard**
   - Deploy a Gradio-based web interface for users to enter search queries, view generated summaries, explore concept maps, and leave manual annotations or comments.

## File Structure