import gradio as gr
import pandas as pd

import time

from fpdf import FPDF  
from bs4 import BeautifulSoup


from util import (
    fetch_arxiv,
    fetch_semantic_scholar,
    fetch_crossref,
    summarize_abstract_spacy,
    extract_entities,
    build_concept_map,
    summarize_abstracts_llm,
    build_global_concept_map
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

def search_papers(query: str, max_results: str):
    try:
        n = int(max_results)
        n = n if n>0 else 5
    except:
        n = 5

    # dispatch based on dropdown
    # if source == "arXiv":
    #     papers = fetch_arxiv(query, max_results=n)
    # elif source == "Semantic Scholar":
    #     papers = fetch_semantic_scholar(query, max_results=n)
    # else: 
    #     papers = fetch_crossref(query, max_results=n)

    records = []
    papers = fetch_arxiv(query, max_results=n)
    # Abstract summaries using Spacy
    for p in papers:
        print(p["title"])
        raw   = summarize_abstract_spacy(p["abstract"], num_sentences=2)
        clean = raw.replace("\n"," ")
        records.append({"Title": p["title"], "Summary": clean})
    df = pd.DataFrame(records)
    return papers, df_to_html_table(df)

def process_all(papers):
    """Process all papers to generate a cross-paper summary and display concept maps."""
    # Cross-Paper Summary using Qwen
    abstracts = [p["abstract"] for p in papers]
    yield "Progress: Generating cross-paper summary...", None
    time.sleep(1)  # Simulate processing time
    narrative = summarize_abstracts_llm(abstracts)

    # Global concept map
    yield "Progress: Building global concept map...", None
    time.sleep(1)  # Simulate processing time
    global_map = build_global_concept_map(papers)
    global_html = global_map.generate_html()
    escaped_global_html = global_html.replace('"', '&quot;')
    iframe_global = (
        '<iframe '
        f'srcdoc="{escaped_global_html}" '
        'style="width:100%; height:1000px; border:none;"'
        '></iframe>'
    )

    # Individual concept maps
    parts = [
        "<div style='width:100%;'>",
        "<h1>Cross-Paper Summary</h1>",
        f"<p>{narrative}</p>",
        "<h1>Global Concept Network</h1>",
        iframe_global,
        "<hr><h1>Per-Paper Concept Maps</h1>"
    ]
    for i, p in enumerate(papers):
        yield f"Progress: Processing paper {i + 1} of {len(papers)} - {p['title']}...", None
        time.sleep(1)  # Simulate processing time
        summary = summarize_abstract_spacy(p["abstract"], num_sentences=3).replace("\n", " ")
        ents = extract_entities(p["abstract"])
        graph = build_concept_map(ents)
        html = graph.generate_html()
        escaped_html = html.replace('"', '&quot;')
        iframe = (
            '<iframe '
            f'srcdoc="{escaped_html}" '
            'style="width:100%; height:1000px; border:none;"'
            '></iframe>'
        )
        parts += [
            f"<h2>{p['title']}</h2>",
            f"<p>{summary}</p>",
            iframe,
            "<hr>"
        ]
    parts.append("</div>")

    # Final output
    final_html = "\n".join(parts)
    yield "Progress: Completed!", final_html
    return final_html

def export_to_pdf(html_content):
    """Export the summary to a PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Add the Summary Section
    pdf.set_font("Arial", style="B", size=14)  # Bold font for headers
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font for content
    pdf.ln(5)  # Add a small vertical space

    for paragraph in soup.find_all("p"):
        pdf.multi_cell(0, 10, paragraph.get_text())
        pdf.ln(5)  # Add spacing between paragraphs

    # Save the PDF to a file
    pdf_file = "summary.pdf"
    pdf.output(pdf_file)
    return pdf_file

with gr.Blocks() as demo:
    gr.Markdown("## Academic Paper Summarizer & Concept-Map Explorer")

    with gr.Row():
        query_input = gr.Textbox(label="Search Papers", placeholder="e.g. adversarial ML")
        count_input = gr.Textbox(label="Number of Papers",  value="5", placeholder="Default is 5")
        search_btn  = gr.Button("Search")

    papers_state = gr.State()
    papers_table = gr.HTML(label="Search Results")
    process_btn  = gr.Button("Generate Concept Maps & Summary")
    progress_label = gr.Textbox(label="Progress", value="Waiting for input...", interactive=False)
    output_html  = gr.HTML(label="Results")
    export_pdf_btn = gr.Button("Export as PDF")

    search_btn.click(
        fn=search_papers,
        inputs=[query_input, count_input],
        outputs=[papers_state, papers_table]
    )
    process_btn.click(
        fn=process_all,
        inputs=papers_state,
        outputs=[progress_label, output_html]
    )

    export_pdf_btn.click(
        fn=export_to_pdf,
        inputs=output_html,
        outputs=gr.File(label="Download PDF")
    )

demo.launch()