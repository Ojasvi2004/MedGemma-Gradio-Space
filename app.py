import os
import torch
import spaces
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-1.5-4b-it"
HF_TOKEN = os.getenv("HF_TOKEN")


processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)


model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)


@spaces.GPU(duration=60)
def generate_findings(inputs):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False  
        )
    return outputs

def analyze_image(image: Image.Image, prompt: str):
    if image is None:
        return "Please upload a clinical image."

    if not prompt or prompt.strip() == "":
        prompt = (
            "Analyze this clinical image systematically.\n"
            "Identify the type of image or anatomical region if possible.\n"
            "Describe the relevant visible anatomical structures and any abnormal findings.\n"
            "For each finding, describe its location, appearance, and relevant characteristics.\n"
            "Mention important normal findings when useful.\n"
            "Provide a concise overall impression.\n"
            "If the image quality or available information is insufficient, clearly state the limitation."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    outputs = generate_findings(inputs)

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response

with gr.Blocks(title="MedGemma 1.5 4B Vision Assistant") as demo:
    gr.Markdown("# 🩻 MedGemma 1.5 4B Multimodal Assistant")
    gr.Markdown("Upload a medical radiograph or scan for clinical extraction and structured findings.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Medical Image (X-Ray / CT / Pathology)")
            user_prompt = gr.Textbox(
                label="Instruction / Clinical Query",
                placeholder="e.g., Analyze this image for pleural effusion or cardiomegaly...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="MedGemma Findings", lines=12)

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_image, user_prompt],
        outputs=output_text,
        api_name="predict"  
    )

if __name__ == "__main__":
    demo.launch()