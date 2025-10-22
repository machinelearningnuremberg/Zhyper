import os
import random
import shutil
import string
import threading
import time
from functools import partial
from typing import List, Optional, Tuple, Dict, Generator, Callable

import torch
import gradio as gr
from peft import PeftModel, PeftConfig, get_model_status
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils.eval_hypermod import gen_and_save_lora


GUIDELINE = """
# Zhyper

This is a chatbot demo of the Zhyper model with Mistral-7B-Instruct-v0.2 as the base model.

## How to use

- You can choose either to use the base model or to generate a LoRA from a task description. The generated LoRA will be applied to the base model on-the-fly.
- If you choose to generate a LoRA, you can enter a task description for the LoRA.
- Click the "Generate LoRA" button.
- You can download the generated LoRA to run locally. To use the LoRA locally, you can use the following code:
```python
import zipfile

from transformers import AutoModelForCausalLM

# change PATH_TO_LORA_ZIP_FILE to the path to the downloaded LoRA zip file
with zipfile.ZipFile(PATH_TO_LORA_ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall("./downloaded_lora/")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = model.load_adapter("./downloaded_lora/")
```
For more details, see [the official PEFT documentation](https://huggingface.co/docs/transformers/main/en/peft?load=load_adapter#load-adapter).

## Tips
The model is trained on a specific description template. You can find training and evaluation task descriptions [here](https://github.com/SakanaAI/text-to-lora/tree/main/tasks). Here are also some examples of task descriptions:

- "This task challenges your problem-solving abilities through mathematical reasoning. You must carefully read each scenario and systematically work through the data to compute the final outcome." (gsm8k)
- "This task focuses on developing algorithms in Python for specific scenarios, such as counting characters, assessing conditions between numbers, or converting integers into a different format. Critical thinking and algorithmic design will be important." (humaneval)
- "In this exercise, you need to read short narratives and discern which person or object fits best within the context of the sentence." (winogrande)
"""

FOOTER = """
⚠️ This model is an experimental prototype and is only available for educational and research and development purposes. It is not suitable for commercial use or in environments where failure can have significant effects (mission-critical environments).
The use of this model is at the user's own risk and its performance and results is not guaranteed in any way.
Sakana AI is not responsible for any direct or indirect loss resulting from using this model, regardless of the outcome.
"""

LORA_MAP = {}
LORA_ID = 1
REQ_ID = 1


def create_generation_config(temperature=0.0, max_new_tokens=1024):
    """Create generation configuration for the model."""
    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }


def apply_chat_template(prompt: List[Dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )


def process_single_request(
    model,
    tokenizer,
    chat_history: List[Dict[str, str]],
    generation_config: Dict,
    lora_path: Optional[str],
) -> Generator[Tuple[str, bool], None, None]:
    """Process a single request and yield tuples of (text_chunk, is_finished)."""

    # Apply LoRA adapter if specified
    if lora_path:
        global LORA_ID, LORA_MAP
        if lora_path not in LORA_MAP:
            LORA_MAP[lora_path] = LORA_ID
            LORA_ID += 1
            print(f"Adding new LoRA: {lora_path} with id {LORA_MAP[lora_path]}")
            model.load_adapter(lora_path, f"lora_{LORA_MAP[lora_path]}")

        print(f"Setting adapter to {f'lora_{LORA_MAP[lora_path]}'}")
        model.set_adapter(f"lora_{LORA_MAP[lora_path]}")
    else:
        print("Disabling adapters")
        model.set_adapter("default")

    print(get_model_status(model))
    chat_prompt = apply_chat_template(chat_history, tokenizer)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # Set up streamer for generation
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(**inputs, streamer=streamer, **generation_config)

    # Start generation in a separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output
    complete_text = ""
    for new_text in streamer:
        complete_text += new_text
        is_finished = not thread.is_alive()
        yield new_text, is_finished


def initialize_model(
    model_dir: str, peft_config: PeftConfig
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel(model, peft_config)
    return model, tokenizer


def bot(
    chat_history: List[Dict[str, str]],
    lora_path: Optional[str],
    model_state: Dict,
) -> Generator[List[Dict[str, str]], None, None]:
    # Process base model request
    global REQ_ID
    # LORA_ID += 1
    REQ_ID += 1

    model = model_state["model"]
    tokenizer = model_state["tokenizer"]
    generation_config = create_generation_config()

    chat_history.append({"role": "assistant", "content": ""})
    for text_chunk, is_finished in process_single_request(
        model,
        tokenizer,
        chat_history,
        generation_config,
        lora_path,
    ):
        chat_history[-1]["content"] += text_chunk
        yield chat_history


def create_lora_controls() -> Tuple[gr.Radio, gr.Column, gr.Textbox, gr.Button]:
    """Create a set of LoRA controls (radio, textbox, button)."""
    with gr.Column():
        mode = gr.Radio(
            choices=["Base model", "Generated LoRA"],
            value="Base model",
            label="Model Mode",
        )

        with gr.Column(visible=False) as gen_col:
            gen_text = gr.Textbox(label="Task Description for LoRA", lines=3)
            gen_btn = gr.Button("Generate LoRA")

    return mode, gen_col, gen_text, gen_btn


def user(user_message: str, history: list):
    return "", history + [{"role": "user", "content": user_message}]


def build_demo():
    """Build a single interface."""
    model_dir = "mistralai/Mistral-7B-Instruct-v0.2"

    with gr.Blocks(
        title="Zhyper", fill_height=True, theme=gr.themes.Ocean()
    ) as demo:
        # Loading indicator for the entire app
        loading_indicator = gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                <div style="text-align: center;">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px; font-size: 16px;">Loading the model, please wait...</p>
                </div>
            </div>
            <style>
                .spinner {
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    border: 5px solid rgba(255, 255, 255, 0.3);
                    border-radius: 50%;
                    border-top-color: #42a5f5;
                    animation: spin 1s ease-in-out infinite;
                }
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            </style>
            """,
            visible=True,
        )

        # Main content - initially hidden until model loads
        main_content = gr.Column(visible=False)

        with main_content:
            gr.Markdown(GUIDELINE)

            # Chat Interface
            with gr.Column() as chat_interface:
                title = gr.Markdown("# Zhyper Chat Interface")

                # LoRA Selection Controls
                controls = create_lora_controls()
                mode, gen_col, gen_text, gen_btn = controls

                # Hidden state to store generated LoRA path
                lora_path = gr.State("")
                generated_lora_path = gr.State("")
                lora_description = gr.State("")

                # Status bar for active LoRA
                with gr.Row(visible=False) as status_bar:
                    with gr.Column():
                        status_label = gr.Markdown("", elem_id="lora-status-label")
                    with gr.Column(min_width=150, scale=0):
                        download_btn = gr.DownloadButton(
                            "📥 Download Active LoRA", visible=False, size="sm"
                        )

                # Define toggle_controls as inner function to access UI elements
                def toggle_controls(choice: str, last_generated_path: str, desc: str):
                    """Toggle visibility of LoRA controls based on mode selection."""
                    if choice == "Base model":
                        return {
                            gen_col: gr.update(visible=False),
                            lora_path: gr.update(value=""),
                            generated_lora_path: last_generated_path,
                            status_bar: gr.update(visible=False),
                            status_label: gr.update(value=""),
                        }
                    else:  # "Generated LoRA"
                        if last_generated_path:
                            # Format the status label to include the description
                            label_text = f"🔄 **Active LoRA:** <span style='background-color: rgba(100, 150, 200, 0.2); padding: 3px 8px; border-radius: 4px; margin-left: 8px; display: inline-block;'>{desc}</span>"
                            return {
                                gen_col: gr.update(visible=True),
                                lora_path: gr.update(value=last_generated_path),
                                generated_lora_path: last_generated_path,
                                status_bar: gr.update(visible=True),
                                status_label: gr.update(value=label_text),
                            }
                        else:
                            return {
                                gen_col: gr.update(visible=True),
                                lora_path: gr.update(value=""),
                                generated_lora_path: "",
                                status_bar: gr.update(visible=False),
                                status_label: gr.update(value=""),
                            }

                # Chat UI
                chat = gr.Chatbot(type="messages", height=800)
                msg = gr.Textbox(label="Message")
                gr.Markdown(FOOTER)

        # States
        model_state = gr.State()
        lora_gen_fn = gr.State()

        # Initialize the model right away on startup
        def initialize():
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            layer_indices = torch.arange(0, 32, dtype=torch.long, device=device)
            checkpoint_path = "trained_Zhyper/Zhyper_demo/hypermod.pt"
            zhyper_dir = os.path.dirname(checkpoint_path)
            (
                args,
                hypermod,
                model,
                tokenizer,
                emb_model,
                emb_tokenizer,
                task_desc_format_fn,
                pooling_fn,
            ) = load_hypermod_checkpoint(checkpoint_path, device)
            peft_config = PeftConfig.from_pretrained(zhyper_dir)
            _gen_and_save_lora = partial(
                gen_and_save_lora,
                model_dir=model_dir,
                device=device,
                layer_indices=layer_indices,
                emb_model=emb_model,
                emb_tokenizer=emb_tokenizer,
                task_desc_format_fn=task_desc_format_fn,
                pooling_fn=pooling_fn,
                hypermod=hypermod,
                model=model
            )
            del model, tokenizer

            # Initialize HF model and tokenizer
            model, tokenizer = initialize_model(model_dir, peft_config)

            # Since the default is now "Generated LoRA", we start with gen_col visible
            return {
                model_state: {"model": model, "tokenizer": tokenizer},
                lora_gen_fn: _gen_and_save_lora,
                loading_indicator: gr.update(visible=False),
                main_content: gr.update(visible=True),
            }

        def generate_lora(
            text: str, gen_fn: Callable
        ) -> Tuple[str, str, str, gr.update, gr.update, gr.update, gr.update]:
            """Generate LoRA from text description and return path with updated download button."""
            try:
                if not text.strip():
                    return (
                        "",
                        "",
                        "",
                        gr.update(visible=False),
                        gr.update(value=""),
                        gr.update(visible=False),
                        gr.update(interactive=True, value="Generate LoRA"),
                    )

                curtime = time.strftime("%Y%m%d-%H%M%S")
                uuid = "".join(
                    [
                        random.choice(string.ascii_letters + string.digits)
                        for _ in range(8)
                    ]
                )
                lora_dir = f"/tmp/gen_lora/{curtime}_{uuid}"
                gen_fn(lora_dir=lora_dir, task_desc=text)
                print(f"Generated LoRA at {lora_dir}")
                # make a zip of the lora dir
                zip_path = f"{lora_dir}"
                shutil.make_archive(zip_path, "zip", lora_dir)

                # Prepare status update with description embedded
                label_text = f"🔄 **Active LoRA:** <span style='background-color: rgba(100, 150, 200, 0.2); padding: 3px 8px; border-radius: 4px; margin-left: 8px; display: inline-block;'>{text}</span>"

                return (
                    lora_dir,
                    lora_dir,
                    text,
                    gr.update(visible=True),
                    gr.update(value=label_text),
                    gr.update(value=f"{zip_path}.zip", visible=True),
                    gr.update(interactive=True, value="Generate LoRA"),
                )
            except Exception as e:
                print(f"Error generating LoRA: {str(e)}")
                return (
                    "",
                    "",
                    "",
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(interactive=True, value="Generate LoRA"),
                )

        # Create loading overlay for mode switching
        def before_mode_change():
            return gr.update(interactive=False)

        def before_generate_lora():
            return gr.update(interactive=False, value="Generating LoRA, please wait...")

        # Event handlers with blocking during state transitions
        mode.change(
            fn=before_mode_change, inputs=None, outputs=[mode], queue=False
        ).then(
            toggle_controls,
            inputs=[mode, generated_lora_path, lora_description],
            outputs=[gen_col, lora_path, generated_lora_path, status_bar, status_label],
        ).then(lambda: gr.update(interactive=True), inputs=None, outputs=[mode])

        gen_btn.click(
            fn=before_generate_lora, inputs=None, outputs=[gen_btn], queue=False
        ).then(
            generate_lora,
            inputs=[gen_text, lora_gen_fn],
            outputs=[
                lora_path,
                generated_lora_path,
                lora_description,
                status_bar,
                status_label,
                download_btn,
                gen_btn,
            ],
        )

        # Initialize on load
        demo.load(
            initialize,
            inputs=None,
            outputs=[model_state, lora_gen_fn, loading_indicator, main_content],
        )

        # Set up message chain
        msg.submit(
            user,
            [msg, chat],
            [msg, chat],
            queue=False,
        ).then(
            bot,
            [chat, lora_path, model_state],
            [chat],
        )

    return demo


if __name__ == "__main__":
    os.environ["TMPDIR"] = "/tmp"
    os.environ["GRADIO_TEMP_DIR"] = "/tmp"
    demo = build_demo()
    demo.queue().launch(allowed_paths=["/tmp/gen_lora"], show_error=True, debug=True)
