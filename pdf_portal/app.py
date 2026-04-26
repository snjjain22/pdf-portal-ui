#!/usr/bin/env python3
"""
Next Level Decor — PDF Pipeline Portal (Background Job Version)
Pipeline runs as a detached subprocess so the browser doesn't time out.
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
import shutil
import threading
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
PORTAL_DIR = Path(__file__).parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(PORTAL_DIR.parent)))
PIPELINE_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "pipeline"
OUTPUTS_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "outputs"
PDFS_DIR = REPO_ROOT / "prompt_based_PDF_extractor" / "PDFs"
JOBS_DIR = OUTPUTS_DIR / ".jobs"
PYTHON = sys.executable

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

STAGE_LABELS = {
    0: "Stage 0 — PDF to Images",
    1: "Stage 1 — Detection (slow on CPU)",
    2: "Stage 2 — Field Mapping (LLM)",
    3: "Stage 3 — SEO Content Generation",
    4: "Stage 4 — S3 Image Upload",
    5: "Stage 5 — Application Images",
    6: "Stage 6 — Matrixify CSV Export",
    7: "Stage 7 — Validate & Clean",
    8: "Stage 8 — QR Code Generation",
}
STAGE_CHOICES = [v for v in STAGE_LABELS.values()]


# ── Job management ──────────────────────────────────────────────────────────

def list_jobs():
    """Return all job IDs sorted by most recent first."""
    if not JOBS_DIR.exists():
        return []
    files = sorted(JOBS_DIR.glob("*.status"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]


def start_pipeline_job(pdf_file, vendor_name, skip_stages_list, generate_qr, limit_pages):
    """Start the pipeline as a background subprocess. Returns immediately."""

    if pdf_file is None:
        return "⚠️  Please upload a PDF first.", gr.update(choices=list_jobs())

    pdf_src = Path(pdf_file.name)
    pdf_dest = PDFS_DIR / pdf_src.name
    shutil.copy(str(pdf_src), str(pdf_dest))

    vendor = (vendor_name or "").strip() or "Next Level Decor"

    skip_nums = []
    for s in (skip_stages_list or []):
        try:
            num = int(s.split("—")[0].strip().replace("Stage ", ""))
            skip_nums.append(num)
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{pdf_src.stem.replace(' ', '_')}_{timestamp}"

    job_log = JOBS_DIR / f"{job_id}.log"
    job_status = JOBS_DIR / f"{job_id}.status"

    run_script = PIPELINE_DIR / "run_full_pipeline.py"
    cmd = [PYTHON, str(run_script), str(pdf_dest), "--vendor", vendor]
    if skip_nums:
        cmd += ["--skip"] + [str(n) for n in skip_nums]
    if generate_qr:
        cmd.append("--qr")
    if limit_pages and int(limit_pages) > 0:
        cmd += ["--limit", str(int(limit_pages))]

    job_status.write_text("running")

    log_f = open(job_log, "w", encoding="utf-8")
    log_f.write(f"🚀 Job: {job_id}\n")
    log_f.write(f"   PDF: {pdf_src.name}\n")
    log_f.write(f"   Vendor: {vendor}\n")
    log_f.write(f"   Skipping stages: {skip_nums or 'none'}\n")
    log_f.write(f"   QR codes: {generate_qr}\n")
    log_f.write(f"   Started: {datetime.now().isoformat()}\n")
    log_f.write(f"   Command: {' '.join(cmd)}\n\n")
    log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(PIPELINE_DIR.parent),
    )

    def wait_and_update():
        process.wait()
        try:
            log_f.close()
        except Exception:
            pass
        if process.returncode == 0:
            job_status.write_text("completed")
        else:
            job_status.write_text(f"failed (exit {process.returncode})")

    threading.Thread(target=wait_and_update, daemon=True).start()

    msg = (
        f"✅  **Job started:** `{job_id}`\n\n"
        f"Go to the **Job Status** tab to view live logs and download results.\n\n"
        f"The pipeline runs in the background — you can close this browser tab "
        f"and the job will continue. Come back anytime to check progress."
    )
    return msg, gr.update(choices=list_jobs(), value=job_id)


def get_job_status(job_id: str):
    """Return status, logs, CSV path, QR PDF path for a job."""
    if not job_id:
        return "Select a job to view status.", "", None, None

    status_file = JOBS_DIR / f"{job_id}.status"
    log_file = JOBS_DIR / f"{job_id}.log"

    status = status_file.read_text().strip() if status_file.exists() else "unknown"
    logs = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else "(no logs)"

    # Try to find pipeline outputs. The pipeline normalizes the PDF name as:
    #   pdf_stem.replace(" ", "_").replace("-", "_").upper() + "_PDF"
    parts = job_id.rsplit("_", 2)
    pdf_stem = parts[0] if len(parts) >= 3 else job_id
    normalized = pdf_stem.replace(" ", "_").replace("-", "_").upper() + "_PDF"
    output_dir = OUTPUTS_DIR / normalized

    csv_path = None
    qr_pdf = None
    if output_dir.exists():
        csvs = list(output_dir.glob("matrixify_*.csv"))
        if csvs:
            csv_path = str(csvs[0])
        qr_dir = output_dir / "qr_codes"
        if qr_dir.exists():
            pdfs = list(qr_dir.glob("qr_sheet_*.pdf"))
            if pdfs:
                qr_pdf = str(pdfs[0])

    # Append output directory diagnostic to the status display
    diagnostic = f"\n\n📂 Looking in: `{output_dir.name}` — exists: {output_dir.exists()}"
    if output_dir.exists():
        contents = sorted([p.name for p in output_dir.iterdir()])[:20]
        diagnostic += f"\n📁 Files: {', '.join(contents)}"

    icon = {"running": "🔄", "completed": "✅"}.get(status.split()[0] if status else "", "❌")
    status_md = f"### {icon}  Status: `{status}`{diagnostic}"

    # Show last 200 lines of log to keep it manageable
    log_lines = logs.splitlines()
    log_tail = "\n".join(log_lines[-200:])

    return status_md, log_tail, csv_path, qr_pdf


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="NLD — PDF Pipeline Portal") as demo:
    gr.Markdown(
        """
        # Next Level Decor — PDF Pipeline Portal
        Upload a vendor catalog PDF. Pipeline runs in the background.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Start a job ──────────────────────────────────────────────
        with gr.TabItem("Start Pipeline"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload Catalog PDF", file_types=[".pdf"])
                    vendor_input = gr.Textbox(
                        label="Vendor / Brand Name",
                        value="Next Level Decor",
                    )
                    skip_input = gr.CheckboxGroup(
                        label="Skip Stages",
                        choices=STAGE_CHOICES,
                        value=["Stage 1 — Detection (slow on CPU)", "Stage 5 — Application Images"],
                        info="Stage 1 (CPU detection) takes 30-60 min for a full catalog. Skip it to use auto-generated SKUs.",
                    )
                    qr_toggle = gr.Checkbox(label="Generate QR Codes (Stage 8)", value=True)
                    limit_input = gr.Number(
                        label="Limit Pages (0 = all, useful for testing)",
                        value=0,
                        minimum=0,
                        precision=0,
                    )
                    run_btn = gr.Button("Start Pipeline Job", variant="primary", size="lg")

                with gr.Column():
                    start_msg = gr.Markdown("Upload a PDF and click **Start Pipeline Job**.")

        # ── Tab 2: View job status / logs / outputs ─────────────────────────
        with gr.TabItem("Job Status"):
            with gr.Row():
                jobs_dropdown = gr.Dropdown(
                    label="Select a Job",
                    choices=list_jobs(),
                    interactive=True,
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            status_display = gr.Markdown("Select a job above.")
            logs_display = gr.Textbox(
                label="Live Logs (last 200 lines)",
                lines=25,
                max_lines=25,
                interactive=False,
                autoscroll=True,
            )
            with gr.Row():
                csv_download = gr.File(label="Matrixify CSV", interactive=False)
                qr_download = gr.File(label="QR Sheet PDF", interactive=False)

        # ── Tab 3: About ────────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown(
                """
                ## How it works

                1. **Start Pipeline** — uploads PDF, kicks off background job, returns instantly
                2. **Job Status** — pick your job, see live logs, download outputs when done

                The pipeline subprocess runs detached — your browser can close. Job state lives on the volume.

                ## Pipeline stages
                | Stage | Name | Time on CPU |
                |-------|------|---|
                | 0 | PDF → Images | ~10s |
                | 1 | Product Detection | **30-60 min** for full catalog |
                | 2 | LLM Field Mapping | ~30s per page (LLM API) |
                | 3 | SEO Content | ~30s per page (LLM API) |
                | 4 | S3 Image Upload | ~5s per product |
                | 5 | Application Images | Calls HF Spaces API |
                | 6 | Matrixify CSV | <10s |
                | 7 | Validate & Clean | <10s |
                | 8 | QR Code Generation | <10s |

                **Tip:** Skip Stage 1 for first runs to validate the pipeline, then enable it for production.
                """
            )

    # ── Wire up events ──────────────────────────────────────────────────────

    run_btn.click(
        fn=start_pipeline_job,
        inputs=[pdf_input, vendor_input, skip_input, qr_toggle, limit_input],
        outputs=[start_msg, jobs_dropdown],
    )

    refresh_btn.click(
        fn=lambda jid: (gr.update(choices=list_jobs(), value=jid), *get_job_status(jid)),
        inputs=jobs_dropdown,
        outputs=[jobs_dropdown, status_display, logs_display, csv_download, qr_download],
    )

    jobs_dropdown.change(
        fn=get_job_status,
        inputs=jobs_dropdown,
        outputs=[status_display, logs_display, csv_download, qr_download],
    )


# ── Launch ──────────────────────────────────────────────────────────────────

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
