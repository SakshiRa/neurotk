"""Thin FastAPI wrapper around NeuroTK CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    if not base:
        raise ValueError("empty filename")
    for ch in base:
        if not (ch.isalnum() or ch in "._-"):
            raise ValueError("invalid filename")
    return base


def _save_uploads(files: list[UploadFile], dest: Path) -> int:
    saved = 0
    for item in files:
        if not item.filename:
            continue
        safe = _sanitize_filename(item.filename)
        if not (safe.lower().endswith(".nii") or safe.lower().endswith(".nii.gz")):
            raise ValueError("only .nii or .nii.gz files are supported")
        with (dest / safe).open("wb") as f:
            shutil.copyfileobj(item.file, f)
        saved += 1
    return saved


def _run_cli(command: list[str], env: Optional[Dict[str, str]] = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CLI failed")


def _copy_reports(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "report.html"):
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing {name}")
        shutil.copy2(src, dest_dir / name)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    mode: str = Form("validate"),
    spacing: str = Form("1 1 1"),
    orientation: str = Form("RAS"),
    images_files: list[UploadFile] = File(default=[]),
    labels_files: list[UploadFile] = File(default=[]),
) -> HTMLResponse:
    if not images_files:
        return PlainTextResponse("No image files provided", status_code=400)

    tmp = tempfile.TemporaryDirectory(prefix="neurotk_webapp_")
    root = Path(tmp.name)
    images_dir = root / "images"
    labels_dir = root / "labels"
    out_dir = root / "out"
    images_dir.mkdir()
    labels_dir.mkdir()
    out_dir.mkdir()

    try:
        _save_uploads(images_files, images_dir)
        label_saved_count = 0
        if labels_files:
            label_saved_count = _save_uploads(labels_files, labels_dir)
        labels_provided = label_saved_count > 0
        if not labels_provided:
            labels_dir = None

        report_json = out_dir / "report.json"
        report_html = out_dir / "report.html"

        preprocess_dir: Optional[Path] = None
        env = None
        if mode == "preprocess":
            preprocess_dir = out_dir / "preprocess"
            preprocess_cmd = [
                "neurotk",
                "preprocess",
                "--images",
                str(images_dir),
                "--out",
                str(preprocess_dir),
                "--spacing",
            ] + spacing.split() + ["--orientation", orientation]
            if labels_provided and labels_dir is not None:
                preprocess_cmd.extend(["--labels", str(labels_dir)])
            _run_cli(preprocess_cmd)
            env = dict(**os.environ)
            env["NEUROTK_PREPROCESS_OUTPUT"] = str(preprocess_dir)
            env["NEUROTK_PREPROCESS_SPACING"] = spacing
            env["NEUROTK_PREPROCESS_ORIENTATION"] = orientation
            env["NEUROTK_PREPROCESS_COPY_METADATA"] = "false"
            env["NEUROTK_LABELS_PROVIDED"] = "true" if labels_provided else "false"
            env["NEUROTK_LABELS_UPLOADED"] = str(label_saved_count)

        validate_cmd = [
            "neurotk",
            "validate",
            "--images",
            str(images_dir),
            "--out",
            str(report_json),
            "--html",
            str(report_html),
        ]
        if labels_provided and labels_dir is not None:
            validate_cmd.extend(["--labels", str(labels_dir)])

        _run_cli(validate_cmd, env=env)

        run_id = uuid.uuid4().hex
        run_output_dir = OUTPUTS_DIR / run_id
        _copy_reports(out_dir, run_output_dir)

        return TEMPLATES.TemplateResponse(
            "result.html",
            {
                "request": request,
                "run_id": run_id,
            },
        )
    except Exception as exc:
        return PlainTextResponse(f"Run failed: {type(exc).__name__}: {exc}", status_code=500)
    finally:
        tmp.cleanup()


@app.get("/download/{run_id}/{filename}")
def download(run_id: str, filename: str):
    safe_name = _sanitize_filename(filename)
    if safe_name not in {"report.json", "report.html"}:
        return PlainTextResponse("File not found", status_code=404)

    path = OUTPUTS_DIR / run_id / safe_name
    if not path.exists():
        return PlainTextResponse("File not found", status_code=404)
    return FileResponse(path, filename=safe_name)


@app.get("/view/{run_id}/report", response_class=HTMLResponse)
def view_report(run_id: str) -> HTMLResponse:
    report_path = OUTPUTS_DIR / run_id / "report.html"
    if not report_path.exists():
        return PlainTextResponse("Report not found", status_code=404)
    html = report_path.read_text(encoding="utf-8")
    header = (
        "<div style=\"position:sticky;top:0;background:#ffffff;border-bottom:1px solid #e5e7eb;"
        "padding:12px 20px;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;"
        "font-size:14px;display:flex;gap:16px;z-index:10\">"
        f"<a href=\"/\" style=\"color:#1f4f82;text-decoration:none;\">Back to landing</a>"
        f"<a href=\"/download/{run_id}/report.html\" style=\"color:#1f4f82;text-decoration:none;\">Download HTML</a>"
        f"<a href=\"/download/{run_id}/report.json\" style=\"color:#1f4f82;text-decoration:none;\">Download JSON</a>"
        "</div>"
    )
    return HTMLResponse(content=header + html, status_code=200)
