"""HTML rendering utilities for validation reports."""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt_vector(values: Iterable[object]) -> str:
    return ", ".join(_escape(v) for v in values)


def _fmt_spacing(mean: List[float], std: List[float]) -> str:
    parts = []
    for m, s in zip(mean, std):
        parts.append(f"{m:.3f} \u00b1 {s:.3f}")
    return ", ".join(parts)


def _orientation_consistency(summary: Dict[str, object]) -> str:
    modal = summary.get("orientation_modal")
    deviation = summary.get("orientation_deviation_count")
    if not modal:
        return "unknown"
    modal_text = _fmt_vector(modal)
    if deviation == 0:
        return f"consistent ({modal_text})"
    return f"mixed (modal: {modal_text})"


def _sorted_files(files: Dict[str, Dict[str, object]]) -> List[Tuple[str, Dict[str, object]]]:
    return sorted(files.items(), key=lambda item: item[0].lower())


def render_html_report(report: Dict[str, object]) -> str:
    summary = report.get("summary", {})
    stats_summary = report.get("stats_summary", {})  # type: ignore[assignment]
    files: Dict[str, Dict[str, object]] = report.get("files", {})  # type: ignore[assignment]
    warnings: List[str] = report.get("warnings", [])  # type: ignore[assignment]
    meta: Dict[str, object] = report.get("meta", {})  # type: ignore[assignment]
    timestamp = meta.get("timestamp")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    version = meta.get("version", "unknown")

    issues_counter: Counter[str] = Counter()
    for _name, entry in files.items():
        issues = entry.get("issues") or []
        issues_counter.update(issues)

    files_rows = []
    for filename, entry in _sorted_files(files):
        image_info = entry.get("image") or {}
        label_info = entry.get("label") or {}
        issues = entry.get("issues") or []
        issue_count = len(issues)
        row_class = "has-issues" if issue_count else ""
        shape = image_info.get("shape")
        spacing = image_info.get("spacing")
        orientation = image_info.get("orientation")
        label_present = "n/a"
        if "present" in label_info:
            label_present = "yes" if label_info.get("present") else "no"
        files_rows.append(
            f"<tr class=\"{row_class}\">"
            f"<td>{_escape(filename)}</td>"
            f"<td>{_escape(_fmt_vector(shape) if shape else 'unknown')}</td>"
            f"<td>{_escape(_fmt_vector(spacing) if spacing else 'unknown')}</td>"
            f"<td>{_escape(_fmt_vector(orientation) if orientation else 'unknown')}</td>"
            f"<td>{_escape(label_present)}</td>"
            f"<td>{issue_count}</td>"
            "</tr>"
        )

    issues_rows = []
    for issue, count in sorted(issues_counter.items()):
        issues_rows.append(
            "<tr>"
            f"<td>{_escape(issue)}</td>"
            f"<td>error</td>"
            f"<td>{count}</td>"
            "</tr>"
        )

    warning_rows = []
    for warning in sorted(warnings):
        warning_rows.append(
            "<tr>"
            f"<td>{_escape(warning)}</td>"
            f"<td>warning</td>"
            f"<td>1</td>"
            "</tr>"
        )

    spacing_mean = summary.get("spacing_mean")
    spacing_std = summary.get("spacing_std")
    spacing_text = "unknown"
    if spacing_mean and spacing_std:
        spacing_text = _fmt_spacing(spacing_mean, spacing_std)

    stats_rows = []
    image_stats = stats_summary.get("image_stats", {}) if isinstance(stats_summary, dict) else {}

    def _fmt_metric_value(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, list):
            return "[" + ", ".join(str(v) for v in value) + "]"
        return str(value)

    def _row(metric: str, block: Dict[str, object]) -> None:
        percentiles = block.get("percentiles", {}) if isinstance(block, dict) else {}
        stats_rows.append(
            "<tr>"
            f"<td>{_escape(metric)}</td>"
            f"<td>{_escape(_fmt_metric_value(block.get('min')))}</td>"
            f"<td>{_escape(_fmt_metric_value(block.get('max')))}</td>"
            f"<td>{_escape(_fmt_metric_value(block.get('mean')))}</td>"
            f"<td>{_escape(_fmt_metric_value(block.get('median')))}</td>"
            f"<td>{_escape(_fmt_metric_value(block.get('stdev')))}</td>"
            f"<td>{_escape(_fmt_metric_value(percentiles.get('p0_5')))}</td>"
            f"<td>{_escape(_fmt_metric_value(percentiles.get('p10')))}</td>"
            f"<td>{_escape(_fmt_metric_value(percentiles.get('p90')))}</td>"
            f"<td>{_escape(_fmt_metric_value(percentiles.get('p99_5')))}</td>"
            "</tr>"
        )

    for key in ["shape", "channels", "spacing", "size_mm", "intensity"]:
        block = image_stats.get(key)
        if isinstance(block, dict):
            _row(key, block)

    files_table_body = (
        "".join(files_rows)
        if files_rows
        else "<tr><td colspan=\"6\">No files found</td></tr>"
    )
    issues_table_body = (
        "".join(warning_rows + issues_rows)
        if (warning_rows or issues_rows)
        else "<tr><td colspan=\"3\">No issues reported</td></tr>"
    )

    stats_table_body = (
        "".join(stats_rows)
        if stats_rows
        else "<tr><td colspan=\\\"10\\\">No stats available</td></tr>"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NeuroTK - Dataset Validation Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f5;
      --card: #ffffff;
      --text: #1f2933;
      --muted: #52606d;
      --accent: #2f5d62;
      --border: #d9d9d4;
      --warn: #b45309;
      --error: #9b1c1c;
    }}
    body {{
      margin: 24px;
      font-family: "Georgia", "Times New Roman", serif;
      background: var(--bg);
      color: var(--text);
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    h1 {{
      font-size: 28px;
      color: var(--accent);
    }}
    h2 {{
      font-size: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
      margin-top: 28px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .card {{
      background: var(--card);
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-top: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 12px;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #eef2f3;
      font-weight: 600;
    }}
    tr.has-issues {{
      background: #fff3f0;
    }}
    .warning {{
      color: var(--warn);
      font-weight: 600;
    }}
    .error {{
      color: var(--error);
      font-weight: 600;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .summary-item {{
      background: #fafafa;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
    }}
    .summary-item span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
  <h1>NeuroTK</h1>
  <div class="meta">Dataset Validation Report</div>
  <div class="meta">Timestamp: {_escape(timestamp)}</div>
  <div class="meta">Version: {_escape(version)}</div>

  <div class="card">
    <h2>Dataset Summary</h2>
    <div class="summary-grid">
      <div class="summary-item"><span>Number of images</span>{_escape(summary.get("num_images", "unknown"))}</div>
      <div class="summary-item"><span>Number of labels</span>{_escape(summary.get("num_labels", "unknown"))}</div>
      <div class="summary-item"><span>Files with issues</span>{_escape(summary.get("files_with_issues", "unknown"))}</div>
      <div class="summary-item"><span>Orientation consistency</span>{_escape(_orientation_consistency(summary))}</div>
      <div class="summary-item"><span>Spacing mean ± std</span>{_escape(spacing_text)}</div>
      <div class="summary-item"><span>Number of cases</span>{_escape(stats_summary.get("n_cases", "unknown") if isinstance(stats_summary, dict) else "unknown")}</div>
      <div class="summary-item"><span>Warnings</span>{_escape(", ".join(sorted(warnings)) if warnings else "none")}</div>
    </div>
  </div>

  <div class="card">
    <h2>Image Stats</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Min</th>
          <th>Max</th>
          <th>Mean</th>
          <th>Median</th>
          <th>Stdev</th>
          <th>P0.5</th>
          <th>P10</th>
          <th>P90</th>
          <th>P99.5</th>
        </tr>
      </thead>
      <tbody>
        {stats_table_body}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>File-Level Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Filename</th>
          <th>Shape</th>
          <th>Spacing</th>
          <th>Orientation</th>
          <th>Label present</th>
          <th>Issue count</th>
        </tr>
      </thead>
      <tbody>
        {files_table_body}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Issues</h2>
    <table>
      <thead>
        <tr>
          <th>Issue type</th>
          <th>Severity</th>
          <th>Count</th>
        </tr>
      </thead>
      <tbody>
        {issues_table_body}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    return html_doc


def write_html_report(report: Dict[str, object], path: Path) -> None:
    html_doc = render_html_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
