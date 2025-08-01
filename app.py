import gradio as gr
import os
import tempfile
import pathlib
import random
import string
import huggingface_hub
from typing import Iterable, List
from gradio_logsview.logsview import Log, LogsView, LogsViewRunner

# Define the CLI command for mergekit-evolve
cli = "mergekit-evolve --strategy pool --wandb --wandb-project mergekit-evolve --wandb-entity arcee-ai"

MARKDOWN_DESCRIPTION = """
# mergekit-evolve-gui
A simple GUI to perform an evolutionary model merge using mergekit-evolve.
Specify a YAML configuration file for evolutionary merging and a Hugging Face token.
"""

def merge(yaml_config: str, hf_token: str, repo_name: str) -> Iterable[List[Log]]:
    runner = LogsViewRunner()

    if not yaml_config:
        yield runner.log("Error: Empty YAML configuration.", level="ERROR")
        return

    if not hf_token:
        yield runner.log("Error: No Hugging Face token provided.", level="ERROR")
        return

    api = huggingface_hub.HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        config_path = tmpdir / "config.yaml"
        config_path.write_text(yaml_config)
        yield runner.log(f"Configuration saved to {config_path}")

        if not repo_name:
            repo_name = f"mergekit-evolve-{''.join(random.choices(string.ascii_lowercase, k=7))}"
            yield runner.log(f"Generated repository name: {repo_name}")

        try:
            yield runner.log(f"Creating repository {repo_name}")
            repo_url = api.create_repo(repo_name, exist_ok=True)
            yield runner.log(f"Repository created: {repo_url}")
        except Exception as e:
            yield runner.log(f"Error creating repository: {e}", level="ERROR")
            return

        full_cli = f"{cli} --storage-path {tmpdir} {config_path}"
        yield from runner.run_command(full_cli.split(), cwd=tmpdir)

        if runner.exit_code != 0:
            yield runner.log("Merge failed.", level="ERROR")
            api.delete_repo(repo_url.repo_id)
            return

        yield runner.log("Model merged successfully. Uploading to Hugging Face.")
        yield from runner.run_python(
            api.upload_folder,
            repo_id=repo_url.repo_id,
            folder_path=tmpdir / "merge",
        )
        yield runner.log(f"Model successfully uploaded to Hugging Face: {repo_url.repo_id}")

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN_DESCRIPTION)

    with gr.Row():
        config = gr.Code(language="yaml", lines=10, label="config.yaml")
        with gr.Column():
            token = gr.Textbox(
                lines=1,
                label="Hugging Face Write Token",
                type="password",
                placeholder="Your Hugging Face token"
            )
            repo_name = gr.Textbox(
                lines=1,
                label="Repository Name",
                placeholder="Optional. Random name will be generated if empty."
            )

    button = gr.Button("Merge", variant="primary")
    logs = LogsView(label="Logs")

    button.click(fn=merge, inputs=[config, token, repo_name], outputs=[logs])

demo.queue().launch()
