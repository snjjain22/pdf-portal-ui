import gradio as gr
import torch
import os
import random
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

hf_token = os.environ.get("HF_TOKEN")

print("Loading Model...")
model_id = "black-forest-labs/FLUX.2-klein-9B"

pipe = Flux2KleinPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

print(f"Model loaded on {device}")


def run_generation(processed_data):
    num_inference_steps = 4
    guidance_scale = 1.0
    width = 768
    height = 768

    generated_images = []

    for prompt_room, prompt_furn, img_input in processed_data:
        if not prompt_room or not prompt_furn or not img_input:
            continue

        seed1 = random.randint(0, 2147483647)
        generator1 = torch.Generator(device).manual_seed(seed1)

        img_room = pipe(
            prompt=prompt_room,
            image=img_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator1,
        ).images[0]

        generated_images.append(img_room)

        seed2 = random.randint(0, 2147483647)
        generator2 = torch.Generator(device).manual_seed(seed2)

        img_furn = pipe(
            prompt=prompt_furn,
            image=img_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator2,
        ).images[0]

        generated_images.append(img_furn)

    return generated_images


def process_api_request(pr1, pf1, i1, pr2, pf2, i2, pr3, pf3, i3, pr4, pf4, i4, pr5, pf5, i5):
    raw_data = [
        (pr1, pf1, i1),
        (pr2, pf2, i2),
        (pr3, pf3, i3),
        (pr4, pf4, i4),
        (pr5, pf5, i5),
    ]

    processed_data = []
    for p_room, p_furn, img_path in raw_data:
        if p_room and p_furn and img_path:
            loaded_img = [load_image(img_path)]
            processed_data.append((p_room, p_furn, loaded_img))
        else:
            processed_data.append((None, None, None))

    return run_generation(processed_data)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Laminate Catalog Batch Generator (Room + Furniture Pairs)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Slot 1")
            pr1 = gr.Textbox(label="Room Prompt 1", lines=1)
            pf1 = gr.Textbox(label="Furniture Prompt 1", lines=1)
            i1 = gr.Image(type="filepath", label="Laminate 1")

            gr.Markdown("### Input Slot 2")
            pr2 = gr.Textbox(label="Room Prompt 2", lines=1)
            pf2 = gr.Textbox(label="Furniture Prompt 2", lines=1)
            i2 = gr.Image(type="filepath", label="Laminate 2")

            gr.Markdown("### Input Slot 3")
            pr3 = gr.Textbox(label="Room Prompt 3", lines=1)
            pf3 = gr.Textbox(label="Furniture Prompt 3", lines=1)
            i3 = gr.Image(type="filepath", label="Laminate 3")

            gr.Markdown("### Input Slot 4")
            pr4 = gr.Textbox(label="Room Prompt 4", lines=1)
            pf4 = gr.Textbox(label="Furniture Prompt 4", lines=1)
            i4 = gr.Image(type="filepath", label="Laminate 4")

            gr.Markdown("### Input Slot 5")
            pr5 = gr.Textbox(label="Room Prompt 5", lines=1)
            pf5 = gr.Textbox(label="Furniture Prompt 5", lines=1)
            i5 = gr.Image(type="filepath", label="Laminate 5")

            generate_btn = gr.Button("Generate 10 Images", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Gallery(
                label="Final Renders (Alternates: Room, Furniture, Room, Furniture...)",
                columns=2,
            )

    generate_btn.click(
        fn=process_api_request,
        inputs=[pr1, pf1, i1, pr2, pf2, i2, pr3, pf3, i3, pr4, pf4, i4, pr5, pf5, i5],
        outputs=image_output,
    )

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
