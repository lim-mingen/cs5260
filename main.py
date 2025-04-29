import gradio as gr
import pandas as pd
import base64
from util import (
    fetch_papers,
    summarize_abstract_spacy,
    extract_entities,
    build_concept_map,
    save_graph_html,
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

def search_papers(query: str, max_results: str="5"):
    # parse number of papers
    try:
        n = int(max_results)
        if n <= 0:
            n = 5
    except:
        n = 5
    papers = fetch_papers(query, max_results=n)

    # build a simple table of Title + 2–3 sentence summary
    records = []
    for p in papers:
        raw   = summarize_abstract_spacy(p["abstract"], num_sentences=3)
        clean = raw.replace("\n", " ")
        records.append({"Title": p["title"], "Summary": clean})
    df = pd.DataFrame(records)
    return papers, df_to_html_table(df)

def process_all(papers):
    """
    For each paper:
      1) generate a 2–3 sentence summary,
      2) build & save PyVis concept-map to output/{entry_id}.html,
      3) read that file and embed it as an <iframe> via base64 data URI.
    """
    parts = ["<div style='width:100%;'>"]
    for p in papers:
        # summary
        raw_sum = summarize_abstract_spacy(p["abstract"], num_sentences=3)
        summary = raw_sum.replace("\n", " ")

        # NER & concept map
        ents  = extract_entities(p["abstract"])
        for phrase in ents:
            print(phrase)
        graph = build_concept_map(ents)

        # save standalone HTML
        html_path = save_graph_html(graph, p["entry_id"])

        # read that HTML & encode as base64
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        iframe = (
            f"<iframe "
            f"src='data:text/html;base64,{b64}' "
            f"style='width:100%; height:600px; border:none;'"
            f"></iframe>"
        )

        # assemble section
        parts.append(f"<h2>{p['title']}</h2>")
        parts.append(f"<p>{summary}</p>")
        parts.append(iframe)
        parts.append("<hr>")
    parts.append("</div>")
    return "\n".join(parts)

with gr.Blocks() as demo:
    gr.Markdown("## Academic Paper Summarizer & Concept-Map Explorer")

    with gr.Row():
        query_input = gr.Textbox(
            label="Search Papers",
            placeholder="e.g. adversarial machine learning"
        )
        count_input = gr.Textbox(
            label="Number of Papers",
            placeholder="Default is 5"
        )
        search_btn  = gr.Button("Search")

    papers_state = gr.State()
    papers_table = gr.HTML(label="Search Results")
    process_btn  = gr.Button("Generate Concept Maps")
    output_html  = gr.HTML(label="Concept Maps & Summaries")

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
