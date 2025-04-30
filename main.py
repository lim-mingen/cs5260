import gradio as gr
import pandas as pd
from util import (
    fetch_arxiv,
    fetch_semantic_scholar,
    fetch_crossref,
    summarize_abstract_spacy,
    extract_entities,
    build_concept_map,
    cluster_abstracts,
    summarize_clusters,
    summarize_abstracts_batch,
    build_narrative,
    build_global_concept_map,
)

def df_to_html_table(df: pd.DataFrame) -> str:
    style = """
    <style>
      .my-table { width: 100%; table-layout: fixed; border-collapse: collapse; }
      .my-table th, .my-table td {
        border: 1px solid #ddd;
        padding: 8px;
        word-wrap: break-word;
        white-space: pre-wrap;
      }
      .my-table th { background-color: #f2f2f2; text-align: left; }
    </style>
    """
    html = [style, '<table class="my-table">']
    html.append("<thead><tr><th>Title</th><th>Summary</th></tr></thead><tbody>")
    for _, row in df.iterrows():
        title   = row["Title"].replace("\n", " ")
        summary = row["Summary"].replace("\n", " ")
        html.append(f"<tr><td>{title}</td><td>{summary}</td></tr>")
    html.append("</tbody></table>")
    return "\n".join(html)

def search_papers(query: str, max_results: str, source: str):
    # parse number of papers
    try:
        n = int(max_results)
        n = n if n>0 else 5
    except:
        n = 5

    # dispatch based on dropdown
    # if source == "arXiv":
    #     papers = fetch_arxiv(query, max_results=n)
    # elif source == "Semantic Scholar":  # Disabling since Semantic Scholar paper's dont have abstracts
    #     papers = fetch_semantic_scholar(query, max_results=n)
    # else:  # Disabling since scraped papers from CrossRef are not useful
    #     papers = fetch_crossref(query, max_results=n)

    records = []
    papers = fetch_arxiv(query, max_results=n)
    for p in papers:
        print(p["title"])
        raw   = summarize_abstract_spacy(p["abstract"], num_sentences=2)
        clean = raw.replace("\n"," ")
        records.append({"Title": p["title"], "Summary": clean})
    df = pd.DataFrame(records)
    return papers, df_to_html_table(df)

def process_all(papers):
    # # 1) Cross‐paper clustering & narrative
    # abstracts = [p["abstract"] for p in papers]
    # clusters  = cluster_abstracts(abstracts, max_clusters=5)
    # cl_sums   = summarize_clusters(clusters)
    # narrative = build_narrative(cl_sums)

    # 1) Single cross‐paper summary over all abstracts
    abstracts = [p["abstract"] for p in papers]
    narrative = summarize_abstracts_batch(
        abstracts,
        max_length=400,
        chunk_size=len(papers)    # tweak this if you have more/fewer papers
    )

    # 2) Global concept map
    global_map   = build_global_concept_map(papers)
    global_html  = global_map.generate_html()
    escaped_global_html = global_html.replace('"', '&quot;')
    iframe_global = (
        '<iframe '
        f'srcdoc="{escaped_global_html}" '
        'style="width:100%; height:600px; border:none;"'
        '></iframe>'
    )

    # 3) Individual concept maps
    parts = [
        "<div style='width:100%;'>",
        "<h1>Cross-Paper Summary</h1>",
        f"<p>{narrative}</p>",
        "<h1>Global Concept Network</h1>",
        iframe_global,
        "<hr><h1>Per-Paper Concept Maps</h1>"
    ]
    for p in papers:
        summary = summarize_abstract_spacy(p["abstract"], num_sentences=3).replace("\n"," ")
        ents    = extract_entities(p["abstract"])
        graph   = build_concept_map(ents)
        html    = graph.generate_html()
        escaped_html = html.replace('"', '&quot;')
        iframe  = (
            '<iframe '
            f'srcdoc="{escaped_html}" '
            'style="width:100%; height:600px; border:none;"'
            '></iframe>'
        )
        parts += [
            f"<h2>{p['title']}</h2>",
            f"<p>{summary}</p>",
            iframe,
            "<hr>"
        ]
    parts.append("</div>")

    return "\n".join(parts)

with gr.Blocks() as demo:
    gr.Markdown("## Academic Paper Summarizer & Concept-Map Explorer")

    with gr.Row():
        query_input = gr.Textbox(label="Search Papers", placeholder="e.g. adversarial ML")
        count_input = gr.Textbox(label="Number of Papers", placeholder="Default is 5")
        search_btn  = gr.Button("Search")

    papers_state = gr.State()
    papers_table = gr.HTML(label="Search Results")
    process_btn  = gr.Button("Generate Concept Maps & Summary")
    output_html  = gr.HTML(label="Results")

    search_btn.click(
        fn=search_papers,
        inputs=[query_input, count_input],
        outputs=[papers_state, papers_table]
    )
    process_btn.click(
        fn=process_all,
        inputs=papers_state,
        outputs=output_html
    )

demo.launch()