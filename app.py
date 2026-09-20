import gradio as gr

# Gradion is a python package that allows you to create web interface without manually writting.

def predict_price(description):
    return f"Product: {description}\nPredicted price: $899"
# f means formatted string that allows you to insert variables inside text.
demo = gr.Interface(
    fn = predict_price,
    inputs = gr.Textbox(
        label = "Product Description",
        placeholder="Enter product description.."
    ),
    outputs=gr.Textbox(label="Prediction"),
    title = "AI product price predictor"
)

demo.launch()