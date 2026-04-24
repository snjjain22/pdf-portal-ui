#!/usr/bin/env python3
"""
Next Level Decor — PDF Pipeline Portal
A web UI for running the PDF catalog extraction pipeline.
"""

# ── Monkey-patch gradio_client to handle boolean schemas (bug in 1.3.x) ─────
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)

_gc_utils.get_type = _safe_get_type

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

def _safe_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)

_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
# ── End monkey-patch ────────────────────────────────────────────────────────

import gradio as gr
import os
import subprocess
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# ── Path setup ──────────────────────────────────────────────────────────────
PORTAL_DIR = Path(__file__).parent

# Support REPO_ROOT env var for Docker/Railway deployments (set to /app)
# Falls back to parent of this file's directory for local dev
REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(PORTAL_DIR.parent)))

PIPELINE_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "pipeline"
OUTPUTS_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "outputs"
PDFS_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "PDFs"
PYTHON = sys.executable

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_LABELS = {
    0: "Stage 0 — PDF to Images",
    1: "Stage 1 — Detection (GroundingDINO + SAM)",
    2: "Stage 2 — Field Mapping (LLM)",
    3: "Stage 3 — SEO Content Generation",
    4: "Stage 4 — S3 Image Upload",
    5: "Stage 5 — Application Images (Satyam's Model)",
    6: "Stage 6 — Matrixify CSV Export",
    7: "Stage 7 — Validate & Clean",
    8: "Stage 8 — QR Code Generation",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def list_past_runs():
    """Return list of completed pipeline output directories."""
    if not OUTPUTS_DIR.exists():
        return []
    runs = sorted(
        [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return [d.name for d in runs]


def get_run_summary(run_name: str) -> dict:
    """Read pipeline run summary from outputs directory."""
    run_dir = OUTPUTS_DIR / run_name
    if not run_dir.exists():
        return {}

    summary = {"run_name": run_name, "files": []}

    csv_files = list(run_dir.glob("matrixify_*.csv"))
    if csv_files:
        summary["csv"] = str(csv_files[0])
        summary["csv_name"] = csv_files[0].name

    qr_dir = run_dir / "qr_codes"
    if qr_dir.exists():
        pdf_files = list(qr_dir.glob("qr_sheet_*.pdf"))
        if pdf_files:
            summary["qr_pdf"] = str(pdf_files[0])
            summary["qr_pdf_name"] = pdf_files[0].name

        manifest = qr_dir / "qr_manifest.csv"
        if manifest.exists():
            summary["manifest"] = str(manifest)

        png_count = len(list(qr_dir.glob("*.png")))
        summary["qr_count"] = png_count

    return summary


# ── Tab 1: Run Pipeline ──────────────────────────────────────────────────────

def run_pipeline(pdf_file, vendor_name, skip_stages_list, generate_qr, limit_pages, progress=gr.Progress()):
    """Run the pipeline and stream logs."""

    if pdf_file is None:
        yield "⚠️  Please upload a PDF first.", None, None, gr.update(choices=list_past_runs())
        return

    # Copy uploaded PDF to PDFs/ directory
    pdf_src = Path(pdf_file.name)
    pdf_dest = PDFS_DIR / pdf_src.name
    shutil.copy(str(pdf_src), str(pdf_dest))

    vendor = vendor_name.strip() or "Next Level Decor"

    # Build skip list
    skip_nums = [int(s.split("—")[0].strip().replace("Stage ", "")) for s in skip_stages_list]
    skip_arg = " ".join(str(s) for s in skip_nums) if skip_nums else ""

    # Build command
    run_script = PIPELINE_DIR / "run_full_pipeline.py"
    cmd = [PYTHON, str(run_script), str(pdf_dest), f"--vendor", vendor]

    if skip_arg:
        cmd += ["--skip"] + [str(s) for s in skip_nums]
    if generate_qr:
        cmd.append("--qr")
    if limit_pages and limit_pages > 0:
        cmd += ["--limit", str(limit_pages)]

    log_lines = []
    log_lines.append(f"🚀  Starting pipeline: {pdf_src.name}")
    log_lines.append(f"    Vendor: {vendor}")
    log_lines.append(f"    Skipping stages: {skip_nums or 'none'}")
    log_lines.append(f"    QR codes: {'yes' if generate_qr else 'no'}")
    log_lines.append(f"    Command: {' '.join(cmd)}\n")
    yield "\n".join(log_lines), None, None, gr.update(choices=list_past_runs())

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(PIPELINE_DIR.parent),
        )

        for line in iter(process.stdout.readline, ""):
            log_lines.append(line.rstrip())
            yield "\n".join(log_lines[-200:]), None, None, gr.update(choices=list_past_runs())

        process.wait()

        if process.returncode == 0:
            log_lines.append("\n✅  Pipeline complete!")
        else:
            log_lines.append(f"\n❌  Pipeline failed (exit code {process.returncode})")

    except Exception as e:
        log_lines.append(f"\n❌  Error: {e}")

    # Determine output dir name
    pdf_stem = pdf_src.stem.replace(" ", "_").replace("-", "_").upper() + "_PDF"
    run_dir = OUTPUTS_DIR / pdf_stem
    summary = get_run_summary(pdf_stem) if run_dir.exists() else {}

    csv_path = summary.get("csv")
    qr_pdf_path = summary.get("qr_pdf")

    yield (
        "\n".join(log_lines[-200:]),
        csv_path,
        qr_pdf_path,
        gr.update(choices=list_past_runs()),
    )


# ── Tab 2: Results ───────────────────────────────────────────────────────────

def load_run_details(run_name: str):
    """Load details for a selected past run."""
    if not run_name:
        return "Select a run to view details.", None, None, "0"

    summary = get_run_summary(run_name)
    if not summary:
        return "No data found for this run.", None, None, "0"

    lines = [f"**Run:** {run_name}"]
    if "csv_name" in summary:
        lines.append(f"**Matrixify CSV:** {summary['csv_name']}")
    if "qr_pdf_name" in summary:
        lines.append(f"**QR Sheet PDF:** {summary['qr_pdf_name']}")
    if "qr_count" in summary:
        lines.append(f"**QR codes generated:** {summary['qr_count']}")

    return (
        "\n".join(lines),
        summary.get("csv"),
        summary.get("qr_pdf"),
        str(summary.get("qr_count", "0")),
    )


def regenerate_qr(run_name: str):
    """Re-run Stage 8 QR generation for an existing run."""
    if not run_name:
        return "⚠️  Select a run first.", None

    run_dir = OUTPUTS_DIR / run_name
    if not run_dir.exists():
        return f"❌  Output directory not found: {run_name}", None

    qr_script = PIPELINE_DIR / "8_generate_qr_codes.py"
    cmd = [PYTHON, str(qr_script), "--output-dir", str(run_dir), "--force"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PIPELINE_DIR.parent))
        output = result.stdout + result.stderr
        summary = get_run_summary(run_name)
        return output, summary.get("qr_pdf")
    except Exception as e:
        return f"❌  Error: {e}", None


# ── UI Layout ────────────────────────────────────────────────────────────────

STAGE_CHOICES = [f"Stage {k} — {v.split('—')[1].strip()}" for k, v in STAGE_LABELS.items()]

with gr.Blocks(title="Next Level Decor — PDF Pipeline Portal") as demo:

    gr.Markdown(
        """
        # Next Level Decor — PDF Pipeline Portal
        Upload a vendor catalog PDF and run the full extraction pipeline.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Run Pipeline ──────────────────────────────────────────────
        with gr.TabItem("Run Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(
                        label="Upload Catalog PDF",
                        file_types=[".pdf"],
                    )
                    vendor_input = gr.Textbox(
                        label="Vendor / Brand Name",
                        placeholder="e.g. Xterio, Greenlam, Virgo",
                        value="Next Level Decor",
                    )
                    skip_input = gr.CheckboxGroup(
                        label="Skip Stages",
                        choices=STAGE_CHOICES,
                        value=[],
                        info="All stages run on CPU. Stage 5 calls HuggingFace Spaces API for image generation.",
                    )
                    with gr.Row():
                        qr_toggle = gr.Checkbox(label="Generate QR Codes (Stage 8)", value=True)
                        limit_input = gr.Number(
                            label="Limit Pages (0 = all)",
                            value=0,
                            minimum=0,
                            precision=0,
                        )
                    run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

                with gr.Column(scale=2):
                    log_output = gr.Textbox(
                        label="Pipeline Logs",
                        lines=30,
                        max_lines=30,
                        autoscroll=True,
                        interactive=False,
                        placeholder="Logs will appear here when the pipeline runs...",
                    )
                    with gr.Row():
                        csv_download = gr.File(label="Download Matrixify CSV", interactive=False)
                        qr_download = gr.File(label="Download QR Sheet PDF", interactive=False)

            past_runs_state = gr.State(list_past_runs())

            run_btn.click(
                fn=run_pipeline,
                inputs=[pdf_input, vendor_input, skip_input, qr_toggle, limit_input],
                outputs=[log_output, csv_download, qr_download, past_runs_state],
            )

        # ── Tab 2: Past Results ──────────────────────────────────────────────
        with gr.TabItem("Results"):
            with gr.Row():
                with gr.Column(scale=1):
                    runs_dropdown = gr.Dropdown(
                        label="Select a Past Run",
                        choices=list_past_runs(),
                        interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh List")
                    regen_qr_btn = gr.Button("Re-generate QR Codes", variant="secondary")

                with gr.Column(scale=2):
                    run_details = gr.Markdown("Select a run to view details.")
                    qr_count_label = gr.Textbox(label="QR Codes", interactive=False)
                    with gr.Row():
                        result_csv = gr.File(label="Matrixify CSV", interactive=False)
                        result_qr = gr.File(label="QR Sheet PDF", interactive=False)
                    regen_log = gr.Textbox(label="Re-generation Log", lines=5, interactive=False)

            refresh_btn.click(
                fn=lambda: gr.update(choices=list_past_runs()),
                outputs=runs_dropdown,
            )

            runs_dropdown.change(
                fn=load_run_details,
                inputs=runs_dropdown,
                outputs=[run_details, result_csv, result_qr, qr_count_label],
            )

            regen_qr_btn.click(
                fn=regenerate_qr,
                inputs=runs_dropdown,
                outputs=[regen_log, result_qr],
            )

        # ── Tab 3: About ─────────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown(
                """
                ## Pipeline Stages

                | Stage | Name | Description |
                |-------|------|-------------|
                | 0 | PDF to Images | Converts PDF pages to high-res JPEGs (600 DPI) |
                | 1 | Detection | Detects product swatches using GroundingDINO + SAM (needs GPU) |
                | 2 | Field Mapping | LLM derives appearance, finish, color, size, thickness |
                | 3 | SEO | Generates titles, descriptions, handles, FAQs |
                | 4 | S3 Upload | Uploads product images to AWS S3 |
                | 5 | App Images | Generates room/furniture lifestyle images via FLUX model (needs GPU) |
                | 6 | Matrixify CSV | Exports Shopify-ready 61-column CSV |
                | 7 | Validate & Clean | Validates CSV integrity |
                | 8 | QR Codes | Generates QR code PNGs + printable PDF sheet |

                ## QR Code System
                QR codes point to `https://go.nextleveldecor.in/{SKU}` which redirects to the Shopify product page.
                This allows QR codes to be printed on product samples **before** the product is live on Shopify.

                ## Image Generator
                Stage 5 calls the HuggingFace Space API (`Automate-GPT/NLDApplicationImageGenerator`) remotely.
                All other stages run on CPU on Railway — no GPU required.
                """
            )

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

try:
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        show_api=False,
        share=False,
        prevent_thread_lock=True,
    )
except ValueError as e:
    print(f"Ignoring Gradio post-launch check: {e}")

import time
print("Gradio server running. Keeping main thread alive...")
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("Shutting down...")
