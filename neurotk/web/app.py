# neurotk/web/app.py
"""Minimal web UI for NeuroTK."""

from __future__ import annotations

import argparse
import atexit
import shutil
import tempfile
import zipfile
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from neurotk import __version__
from neurotk import cli as neurotk_cli
from neurotk.preprocess import preprocess_dataset


MAX_TOTAL_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_FILE_BYTES = 250 * 1024 * 1024
ALLOWED_EXTENSIONS = (".nii", ".nii.gz", ".zip")
ALLOWED_DATA_EXTENSIONS = (".nii", ".nii.gz")


templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)


class RunState:
    def __init__(self) -> None:
        self.tempdir: Optional[tempfile.TemporaryDirectory] = None
        self.root: Optional[Path] = None
        self.report_json: Optional[Path] = None
        self.report_html: Optional[Path] = None
        self.preprocess_dir: Optional[Path] = None
        self.preprocess_zip: Optional[Path] = None

    def cleanup(self) -> None:
        if self.tempdir is not None:
            self.tempdir.cleanup()
        self.tempdir = None
        self.root = None
        self.report_json = None
        self.report_html = None
        self.preprocess_dir = None
        self.preprocess_zip = None


_run_state = RunState()
_run_lock = Lock()


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    if not base:
        raise ValueError("empty filename")
    for ch in base:
        if not (ch.isalnum() or ch in "._-"):
            raise ValueError("invalid filename")
    if not base.lower().endswith(ALLOWED_EXTENSIONS):
        raise ValueError("unsupported file type")
    return base


def _ensure_allowed_data_file(name: str) -> None:
    if not name.lower().endswith(ALLOWED_DATA_EXTENSIONS):
        raise ValueError("unsupported dataset file")


def _write_uploadfile(upload: UploadFile, dest: Path, remaining: int) -> int:
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES or total > remaining:
                raise ValueError("upload exceeds size limit")
            f.write(chunk)
    return total


def _extract_zip(zip_path: Path, dest_dir: Path, remaining: int) -> int:
    total = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if Path(name).name != name:
                raise ValueError("nested paths are not allowed")
            safe_name = _sanitize_filename(name)
            _ensure_allowed_data_file(safe_name)
            if info.file_size > MAX_FILE_BYTES or info.file_size > remaining:
                raise ValueError("upload exceeds size limit")
            total += info.file_size
            if total > remaining:
                raise ValueError("upload exceeds size limit")
            with zf.open(info) as src, (dest_dir / safe_name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return total


def _save_uploads(
    uploads: List[UploadFile],
    dest_dir: Path,
    remaining: int,
) -> int:
    total = 0
    for upload in uploads:
        if not upload.filename:
            continue
        safe_name = _sanitize_filename(upload.filename)
        if safe_name.lower().endswith(".zip"):
            zip_path = dest_dir / safe_name
            used = _write_uploadfile(upload, zip_path, remaining - total)
            total += used
            used = _extract_zip(zip_path, dest_dir, remaining - total)
            total += used
            zip_path.unlink(missing_ok=True)
        else:
            _ensure_allowed_data_file(safe_name)
            used = _write_uploadfile(upload, dest_dir / safe_name, remaining - total)
            total += used
    return total


def _has_data_files(path: Path) -> bool:
    return any(p.is_file() for p in path.iterdir())


def _prepare_run_workspace() -> Path:
    _run_state.cleanup()
    _run_state.tempdir = tempfile.TemporaryDirectory(prefix="neurotk_web_")
    _run_state.root = Path(_run_state.tempdir.name)
    return _run_state.root


def _run_validation(images_dir: Path, labels_dir: Optional[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_html = out_dir / "report.html"
    args = argparse.Namespace(
        images=images_dir,
        labels=labels_dir,
        out=report_json,
        max_samples=None,
        html=report_html,
        command="validate",
    )
    neurotk_cli._run_validate(args)
    _run_state.report_json = report_json
    _run_state.report_html = report_html


def _run_preprocess(
    images_dir: Path,
    labels_dir: Optional[Path],
    out_dir: Path,
    spacing: Tuple[float, float, float],
    orientation: str,
    copy_metadata: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        spacing=spacing,
        orientation=orientation,
        dry_run=False,
        copy_metadata=copy_metadata,
    )
    _run_state.preprocess_dir = out_dir


def _build_preprocess_zip() -> Optional[Path]:
    if _run_state.preprocess_dir is None or _run_state.root is None:
        return None
    if _run_state.preprocess_zip is not None and _run_state.preprocess_zip.exists():
        return _run_state.preprocess_zip
    zip_path = _run_state.root / "preprocess_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in _run_state.preprocess_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(_run_state.preprocess_dir))
    _run_state.preprocess_zip = zip_path
    return zip_path


def _extract_report_body(report_path: Path) -> str:
    text = report_path.read_text(encoding="utf-8")
    lower = text.lower()
    start = lower.find("<body")
    if start == -1:
        return text
    start = lower.find(">", start)
    end = lower.rfind("</body>")
    if start == -1 or end == -1:
        return text
    return text[start + 1 : end].strip()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = {"request": request, "version": __version__}
    return templates.TemplateResponse("index.html", context)


@app.post("/run")
def run(
    request: Request,
    mode: str = Form("validate"),
    spacing: str = Form("1 1 1"),
    orientation: str = Form("RAS"),
    copy_metadata: Optional[bool] = Form(None),
    images_files: List[UploadFile] = File(default=[]),
    labels_files: List[UploadFile] = File(default=[]),
) -> RedirectResponse:
    with _run_lock:
        workspace = _prepare_run_workspace()
        images_dir = workspace / "images"
        labels_dir = workspace / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        total_used = 0
        try:
            total_used += _save_uploads(images_files, images_dir, MAX_TOTAL_UPLOAD_BYTES)
            total_used += _save_uploads(
                labels_files, labels_dir, MAX_TOTAL_UPLOAD_BYTES - total_used
            )
        except Exception as exc:
            return PlainTextResponse(
                f"Upload failed: {type(exc).__name__}: {exc}", status_code=400
            )

        if not _has_data_files(images_dir):
            return PlainTextResponse("No image files provided", status_code=400)

        if not _has_data_files(labels_dir):
            labels_dir = None

        try:
            _run_validation(images_dir, labels_dir, workspace / "reports")
        except Exception as exc:
            return PlainTextResponse(
                f"Validation failed: {type(exc).__name__}: {exc}", status_code=500
            )

        if mode == "preprocess":
            try:
                parts = [float(x) for x in spacing.split()]
                if len(parts) != 3:
                    raise ValueError("spacing must have 3 values")
                spacing_tuple = (parts[0], parts[1], parts[2])
                _run_preprocess(
                    images_dir,
                    labels_dir,
                    workspace / "preprocess",
                    spacing_tuple,
                    orientation.strip().upper(),
                    copy_metadata is True,
                )
            except Exception as exc:
                return PlainTextResponse(
                    f"Preprocess failed: {type(exc).__name__}: {exc}",
                    status_code=500,
                )

    return RedirectResponse(url="/report", status_code=303)


@app.get("/report", response_class=HTMLResponse)
def report(request: Request) -> HTMLResponse:
    if _run_state.report_html is None or _run_state.report_json is None:
        return RedirectResponse(url="/", status_code=303)
    report_body = _extract_report_body(_run_state.report_html)
    context = {
        "request": request,
        "report_body": report_body,
        "has_preprocess": _run_state.preprocess_dir is not None,
    }
    return templates.TemplateResponse("report.html", context)


@app.get("/download/{file_name}")
def download(file_name: str):
    if _run_state.report_json is None or _run_state.report_html is None:
        return PlainTextResponse("No report available", status_code=404)

    if file_name == "report.json":
        return FileResponse(_run_state.report_json, filename="report.json")
    if file_name == "report.html":
        return FileResponse(_run_state.report_html, filename="report.html")
    if file_name == "preprocess.zip":
        zip_path = _build_preprocess_zip()
        if zip_path is None:
            return PlainTextResponse("No preprocess outputs", status_code=404)
        return FileResponse(zip_path, filename="preprocess_outputs.zip")
    return PlainTextResponse("File not found", status_code=404)


@atexit.register
def _cleanup_on_exit() -> None:
    _run_state.cleanup()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("neurotk.web.app:app", host="127.0.0.1", port=8000, reload=False)
