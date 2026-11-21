import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from deals import Deal
from sample_deals import SAMPLE_DEALS

# Read variables from .env if the file exists
load_dotenv()

rag = None

def get_rag():
    global rag
    if rag is None:
        from rag_engine import RAGPricePredictor 
        rag = RAGPricePredictor()

    return rag

def load_sample_deals():
    rows = []
    for item in SAMPLE_DEALS:
        # Dictionary unpacking (**item): see Notes/app.py.md, Topic 1.
        deal = Deal(**item)
        rows.append([deal.product_description, deal.price,deal.url])
    status = f"Loaded {len(rows)} sample deals. These are examples, not live offers."
    return status, rows

def load_live_deals():
    try:
        from rss_deals import fetch_rss_deals
        deals, warnings = fetch_rss_deals()
        rows = [
            [deal["source"], deal["title"], deal["summary"],deal["url"]]
            for deal in deals 
        ]
        status = f"Loaded {len(rows)} Live RSS entries. Prices have not been extracted yet."
        if warnings:
            status += "\n" + "\n".join(warnings)
        return status,rows
    except Exception as error:
        return f"Could not load live RSS entries: {error}",[]


def format_similar_products(products):
    lines = []
    for number, item in enumerate(products, start = 1):
        lines.append(
            f"{number}.{item['description']}\n"
            f" Price: ${item['price']:.2f} | "
            f"Category: {item['category']}\n"
        )
        # print("lines",lines)
        # print("\n".join(lines))
        # raise RuntimeError("DEBUG STOP: Lines printed successfully")
    return "\n".join(lines)


# Gradion is a python package that allows you to create web interface without manually writting.

def predict_price(description,mode):
    #raise RuntimeError(description,"User Description")
    if not description or not description.strip():
        return "Please enter a product description.", ""

    if mode == "Begineer Ensemble":
        return "This mode is not implemented in your uploaded project yet.",""
    if mode not in ("RAG only (begineer)","RAG + Frontier LLM"):
        return "Please select a supported prediction mode.",""
    try:   
        local_price, similar_products = get_rag.estimate_price_locally(description.strip())
    except Exception as error:
        return f"Price prediction could not run: {error}",""    
    #raise RuntimeError(local_price,similar_products)
    retrieved_text = format_similar_products(similar_products)
    #raise RuntimeError(retrieved_text)
# f means formatted string that allows you to insert variables inside text.

# MODE 1:
    if mode == "RAG only (begineer)":
        result = (
            f"Estimated price: ${local_price:.2f}"
        )
        return result,retrieved_text
    if mode == "RAG + Frontier LLM":
        try:
            from frontier_agent import predict_with_frontier_llm

            frontier_price = predict_with_frontier_llm(description.strip(),similar_products)
          
            return f"Estimated price: ${frontier_price:.2f}",retrieved_text
        except Exception as error:
            return (
                f"Reason: {error}", retrieved_text
            )
                                                                               
with gr.Blocks() as demo:

    description = gr.Textbox(
        label = "Product Description",
        placeholder= "Example: Shure professional USB podcast microphone",
        lines = 4
    )

    mode = gr.Radio(
        choices= [
            "RAG only (begineer)",
            "RAG + Frontier LLM",
            "Begineer Ensemble"
        ],
        value= "RAG only (begineer)",
        label="Prediction Mode"
    )

    predict_button = gr.Button("Predict Price")
    prediction = gr.Textbox(
        label = "Prediction",
        lines = 6
    )
    similar_products = gr.Textbox(
        label = "5 similar products found by RAG",
        lines = 14,
    )
    predict_button.click(
        fn = predict_price,
        inputs = [description,mode],
        outputs= [prediction, similar_products]
    )
    with gr.Tab("Sample Deals"):
        load_button = gr.Button("Load Sample Deals", variant="primary")
        deal_status = gr.Textbox(label="Status", interactive=False)
        deals_table = gr.DataFrame(
            headers=["Product description", "Sample price (USD)","Example URL"],
            datatype= ["str","number","str"],
            value=[],
            # Interactive controls whether users can edit the table directly in your app.
            # interactive = true means users can view and edit the table's cells, false means cannot.
            interactive=False,
            # wrap means long text automatically continues onto the next line inside the same table cell.
            wrap=True,    
        )
        load_button.click( 
            fn = load_sample_deals,
            inputs=[],
            outputs=[deal_status,deals_table]
        )
    with gr.Tab("Live RSS"):
        live_button = gr.Button("load live RSS Deals",variant="primary")
        live_status = gr.Textbox(label="RSS status",lines = 3, interactive= False)
        live_table = gr.DataFrame( 
            headers=["Source","Product title","RSS summary","Deal URL"],
            datatype= ["str","str","str","str"],
            value=[],
            interactive=False,
            wrap= True,
        )
        live_button.click(
            fn = load_live_deals,
            inputs=[],
            outputs=[live_status, live_table],
        )


demo.launch()