import os
import re

def _extract_number(text):
    """Turn a reply such as '$349.99' into the float 349.99."""
    cleaned = text.replace("$", "").replace(",","")
    match = re.search(r"[-+]?\d*\.\d+|\d+",cleaned)

    if not match:
        raise ValueError("The model did not return a price")
    
    return float(match.group())

    #raise RuntimeError("DEBUG STOP: stopped after cleaned and match")

def predict_with_frontier_llm(description, similar_products):
    print(description,similar_products)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Use 'RAG only' mode first or add your API  key to a .env file."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is not installed. Run: pip install openai"
        ) from error
    
    context = ""
    for item in similar_products:
        context += (
            f"\nSimilar product: {item['description']}\n"
            f"Known price: ${item['price']:.2f}\n"
        )

    prompt = f"""
Estimate the price of the product below.

Product:
{description}

Here are similar products retrieved from our vector database:
{context}


"""
    #raise RuntimeError(context,"Context")

    client = OpenAI()

    # You can change this in .env:
    # OPENAI_MODEL=gpt-5.6-luna
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "none"},
    )

    # here output_text is defined by the OpenAI SDK. it is the nonchangeable parameter of response.
    #raise RuntimeError(response.output_text,"response")
    return round(_extract_number(response.output_text),2)
