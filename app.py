import gradio as gr


def test_project():
    return "Hello"


with gr.Blocks(title="PricePredictor") as demo:


    button = gr.Button("Click Me ")
    output = gr.Textbox(label="Output")

    button.click(
        fn=test_project,
        outputs=output
    )


if __name__ == "__main__":
    demo.launch()