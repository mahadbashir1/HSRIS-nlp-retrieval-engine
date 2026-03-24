import torch
import gradio as gr
import pandas as pd
import plotly.io as pio
import os
from evaluate import load_inference_assets

print("Deploying Inference Engine... Loading indices...")
engine, _, _ = load_inference_assets()
print("Engine ready.")

def infer_ticket_type(query: str, beta: float) -> str:
    hits = engine.retrieve(query, top_n=5, beta=beta)
    return hits['Ticket Type'].value_counts().index[0]

def build_result_card(row: pd.Series, position: int) -> str:
    resolution_snippet = str(row['Resolution'])[:280]
    return (
        f"Ticket Match #{position}\n\n"
        f"Category   : {row['Ticket Type']}\n"
        f"Urgency    : {row['Ticket Priority']}\n"
        f"Medium     : {row['Ticket Channel']}\n"
        f"Subject    : {row['Ticket Subject']}\n\n"
        f"Resolution :\n{resolution_snippet}..."
    )

def run_retrieval(ticket_text: str, beta: float):
    if not ticket_text.strip():
        return 'No input provided.', '', '', ''
    predicted_type = infer_ticket_type(ticket_text, beta)
    top3           = engine.retrieve(ticket_text, top_n=3, beta=beta)
    
    if beta >= 0.65: mode_desc = 'TF-IDF Keyword Focus'
    elif beta <= 0.35: mode_desc = 'GloVe Semantic Focus'
    else: mode_desc = 'Balanced Hybrid Mode'
        
    summary = (
        f"Predicted Category: {predicted_type}\n"
        f"Search Strategy   : {mode_desc}\n"
        f"Beta Value        : {beta:.2f}  "
        f"({beta:.0%} keyword  /  {1-beta:.0%} semantic)"
    )
    
    c1 = build_result_card(top3.iloc[0], 1) if len(top3) > 0 else ''
    c2 = build_result_card(top3.iloc[1], 2) if len(top3) > 1 else ''
    c3 = build_result_card(top3.iloc[2], 3) if len(top3) > 2 else ''
    return summary, c1, c2, c3

DEMO_QUERIES = [
    ['I was billed twice for my subscription this past month.', 0.25],
    ['The software crashes immediately when I try to open it.', 0.5],
    ['Please terminate my account and send my money back.', 0.45],
    ['The product arrived damaged and will not turn on.', 0.55],
    ['I lost all my saved files installing the new patch.', 0.65],
]

cyber_theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Courier Prime"), "Courier New", "monospace"],
    primary_hue="emerald",
    secondary_hue="purple",
    neutral_hue="slate"
).set(
    body_background_fill="#000000",
    body_text_color="#00ff99",
    background_fill_primary="#0a0a0a",
    background_fill_secondary="#000000",
    border_color_accent="#bf00ff",
    border_color_primary="#00ff99",
    block_background_fill="#0a0a0a",
    block_border_width="2px",
    block_border_color="#00ff99",
    button_primary_background_fill="#00ff99",
    button_primary_text_color="#000000",
    button_secondary_background_fill="#000000",
    button_secondary_text_color="#bf00ff",
    button_secondary_border_color="#bf00ff",
    panel_background_fill="#0f0f0f",
    input_background_fill="#1a1a1a",
    slider_color="#00ff99"
)

custom_css = """
.gradio-container { border: 4px dashed #00ff99; box-shadow: inset 0 0 25px rgba(191, 0, 255, 0.5); border-radius: 0px; }
h1, h2, h3 { color: #00ff99 !important; font-family: 'Courier New', monospace; letter-spacing: 3px; font-weight: bolder; }
.output-markdown { color: #bf00ff; border-left: 5px solid #00ff99; padding-left: 10px; margin-bottom: 10px; }
textarea, input { color: #ffffff !important; font-family: monospace; border: 1px solid #bf00ff !important; border-radius: 0 !important; }
button { text-transform: uppercase; font-family: monospace; font-weight: 900; border-radius: 0 !important; box-shadow: 4px 4px 0px #bf00ff; transition: 0.1s; border: none; }
button:active { transform: translate(2px, 2px); box-shadow: 2px 2px 0px #bf00ff; }
footer { display: none !important; }
"""

eda_path = os.path.join('models', 'dataset_overview.json')
eda_fig = pio.read_json(eda_path) if os.path.exists(eda_path) else None

bench_path = os.path.join('models', 'benchmark_plot.json')
bench_fig = pio.read_json(bench_path) if os.path.exists(bench_path) else None

precision_path = os.path.join('models', 'precision_plot.json')
precision_fig = pio.read_json(precision_path) if os.path.exists(precision_path) else None

comparison_path = os.path.join('models', 'comparison_plot.json')
comparison_fig = pio.read_json(comparison_path) if os.path.exists(comparison_path) else None

with gr.Blocks(title='AI TICKET RETRIEVAL DATABASE', theme=cyber_theme, css=custom_css) as app:

    gr.Markdown("# // AI TICKET RETRIEVAL DATABASE //\n**[Semantic Core & Statistical Frequency Engine]**")
    
    with gr.Tabs():
        with gr.TabItem("Terminal Search Node"):
            gr.Markdown("> INIT SEQUENCE: Enter a customer complaint matrix below to retrieve linked historical incidents.\n"
                        "> OVERRIDE SLIDER: Tune search priority between `0.0` (Neural Embeddings) and `1.0` (Statistical Keyword Mapping).")

            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(label='New Ticket Description', lines=5)
                    beta_control = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.05, label='beta (keyword weight)')
                    with gr.Row():
                        run_btn   = gr.Button('Run Retrieval', variant='primary')
                        reset_btn = gr.Button('Reset', variant='secondary')

                with gr.Column(scale=2):
                    summary_out = gr.Textbox(label='Prediction Summary', lines=5, interactive=False)

            gr.Markdown('### [SYS.LOG] Top 3 Historical Database Matches')
            with gr.Row():
                result_1 = gr.Textbox(label='Terminal Output 1', lines=8, interactive=False)
                result_2 = gr.Textbox(label='Terminal Output 2', lines=8, interactive=False)
                result_3 = gr.Textbox(label='Terminal Output 3', lines=8, interactive=False)

            gr.Examples(examples=DEMO_QUERIES, inputs=[text_input, beta_control])

        with gr.TabItem("Exploratory Data Analysis"):
            gr.Markdown("### > COMPILING CORPUS METRICS...")
            gr.Plot(value=eda_fig, show_label=False)

        with gr.TabItem("Hardware Benchmarks"):
            gr.Markdown("### > DUAL-GPU SIMILARITY DIAGNOSTICS")
            gr.Plot(value=bench_fig, show_label=False)

        with gr.TabItem("Evaluation Report"):
            gr.Markdown("### > QUANTITATIVE ANALYSIS: PRECISION@5 PERFORMANCE")
            gr.Plot(value=precision_fig, show_label=False)
            gr.Markdown("> *Precision@5 indicates the accuracy of retrieving the exact Ticket Type among the top 5 results.*")
            
            gr.Markdown("### > QUALITATIVE EXAMPLES: GLOVE OUTPERFORMING TF-IDF")
            gr.Plot(value=comparison_fig, show_label=False)
            gr.Markdown("""
| User Query (New Ticket) | TF-IDF Keyword Match (Literal) | GloVe Semantic Match (Contextual) | Reason for Outperformance |
| :--- | :--- | :--- | :--- |
| *"My display is shattered."* | *"Display settings missing"* | *"Screen is cracked and broken"* | GloVe understands that 'display' is equivalent to 'screen', and 'shattered' aligns with 'cracked'. |
| *"I want my money back."* | *"How to get back to homepage"* | *"Requesting a refund for recent purchase"* | GloVe natively maps the phrase "money back" to the concept of a "refund". |
| *"The application keeps freezing."* | *"Freezing temperatures affecting delivery"* | *"Software crashes and hangs repeatedly"* | GloVe assumes "application freezing" relates to software crashes, ignoring the literal weather definitions of "freezing". |
| *"Need assistance configuring the router."* | *"Router LED is blinking red"* | *"Help with internet gateway setup"* | GloVe maps "assistance" to "help", "configuring" to "setup", and "router" to "gateway". |
| *"Can I speak to a human?"* | *"Human resources contact info"* | *"Connect me to a live agent or representative"* | GloVe skips literal 'HR' overlaps and links the request for a "human" to a "live agent". |
            """)

    run_btn.click(fn=run_retrieval, inputs=[text_input, beta_control],
                  outputs=[summary_out, result_1, result_2, result_3])
    reset_btn.click(fn=lambda: ('', 0.4, '', '', '', ''), inputs=[],
                    outputs=[text_input, beta_control, summary_out, result_1, result_2, result_3])

if __name__ == '__main__':
    app.launch(share=False, debug=False)
