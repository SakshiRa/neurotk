"""Thin FastAPI wrapper around NeuroTK CLI."""

from __future__ import annotations

import os
import time
import shutil
import subprocess
import tempfile
import uuid
import fcntl
from pathlib import Path
from typing import Optional, Dict, TextIO

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_ROOT = BASE_DIR / "tmp_runs"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
TMP_TTL_MIN = int(os.environ.get("TMP_TTL_MIN", "60"))
RUN_LOCK_FILE = ".run.lock"

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    if not base:
        raise ValueError("empty filename")
    for ch in base:
        if not (ch.isalnum() or ch in "._-"):
            raise ValueError("invalid filename")
    return base


def _save_uploads(files: list[UploadFile], dest: Path, max_bytes: int) -> int:
    saved = 0
    total_bytes = 0
    for item in files:
        if not item.filename:
            continue
        safe = _sanitize_filename(item.filename)
        if not (safe.lower().endswith(".nii") or safe.lower().endswith(".nii.gz")):
            raise ValueError("only .nii or .nii.gz files are supported")
        with (dest / safe).open("wb") as f:
            while True:
                chunk = item.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError("upload exceeds size limit")
                f.write(chunk)
        saved += 1
    return saved, total_bytes


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


def _cleanup_tmp_runs() -> None:
    if TMP_TTL_MIN <= 0:
        return
    cutoff = TMP_TTL_MIN * 60
    now = int(time.time())
    for path in TMP_ROOT.iterdir():
        if not path.is_dir():
            continue
        try:
            age = now - int(path.stat().st_mtime)
        except OSError:
            continue
        if age > cutoff:
            lock_path = path / RUN_LOCK_FILE
            if lock_path.exists():
                try:
                    with lock_path.open("a+") as lock_fh:
                        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, BlockingIOError):
                    # Another request still owns this temp directory.
                    continue
            shutil.rmtree(path, ignore_errors=True)


def _acquire_run_lock(run_dir: Path) -> TextIO:
    lock_fh = (run_dir / RUN_LOCK_FILE).open("a+")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    return lock_fh


def _release_run_lock(lock_fh: TextIO) -> None:
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fh.close()


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

    _cleanup_tmp_runs()
    tmp = tempfile.TemporaryDirectory(prefix="neurotk_webapp_", dir=TMP_ROOT)
    root = Path(tmp.name)
    run_lock_fh = _acquire_run_lock(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    out_dir = root / "out"
    images_dir.mkdir()
    labels_dir.mkdir()
    out_dir.mkdir()

    try:
        _, used_bytes = _save_uploads(images_files, images_dir, MAX_UPLOAD_BYTES)
        label_saved_count = 0
        if labels_files:
            label_saved_count, label_bytes = _save_uploads(
                labels_files, labels_dir, MAX_UPLOAD_BYTES - used_bytes
            )
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
        _release_run_lock(run_lock_fh)
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
