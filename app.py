import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from rag_engine import RAGPricePredictor

# Read variables from .env if the file exists
load_dotenv()

rag = RAGPricePredictor()

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
    if not description or not description.strip():
        return "Please enter a product description."

    local_price, similar_products = rag.estimate_price_locally(description.strip())


    retrieved_text = format_similar_products(similar_products)
# f means formatted string that allows you to insert variables inside text.

with gr.Blocks() as demo:

    description = gr.Textbox(
        label = "Product Description",
        placeholder= "Example: Shure professional USB podcast microphone",
        lines = 4
    )

    mode = gr.Radio(
        choices= [
            "RAG only (beginner)",
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

demo.launch()