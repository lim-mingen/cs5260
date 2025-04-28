import gradio as gr
from util import fetch_papers, summarize_text, extract_entities, build_concept_map


def process_query(query):
    """
    Given a search query, fetch papers, summarize and build concept maps.
    Returns a list of dicts for display.
    """
    papers = fetch_papers(query)
    results = []
    for idx, p in enumerate(papers):
        summary = summarize_text(p['abstract'])
        entities = extract_entities(p['abstract'])
        net = build_concept_map(entities)
        html_file = f"graph_{idx}.html"
        net.save_graph(html_file)
        results.append({
            'title': p['title'],
            'summary': summary,
            'entities': entities,
            'concept_map': html_file,
            'url': p['url']
        })
    return results

# Build Gradio app
demo = gr.Blocks()
with demo:
    gr.Markdown("# Academic Paper Summarization & Visualization")
    query_input = gr.Textbox(label="Enter search query")
    run_button = gr.Button("Analyze Papers")
    output = gr.JSON(label="Results")
    run_button.click(fn=process_query, inputs=query_input, outputs=output)

if __name__ == "__main__":
    demo.launch()