"""Write a table as CSV, Markdown, and LaTeX."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_table(frame: pd.DataFrame, stem: str, csv_dir: Path, paper_dir: Path | None = None) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    out.to_csv(csv_dir / f"{stem}.csv", index=False)
    markdown = _markdown(out)
    (csv_dir / f"{stem}.md").write_text(markdown)
    latex = _latex(out, stem)
    (csv_dir / f"{stem}.tex").write_text(latex)
    if paper_dir is not None:
        out.to_csv(paper_dir / f"{stem}.csv", index=False)
        (paper_dir / f"{stem}.md").write_text(markdown)
        (paper_dir / f"{stem}.tex").write_text(latex)


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def _latex(frame: pd.DataFrame, stem: str) -> str:
    align = "l" * len(frame.columns)
    lines = [
        f"% {stem}",
        "\\begin{tabular}{" + align + "}",
        "\\hline",
        " & ".join(_tex_escape(col) for col in frame.columns) + " \\\\",
        "\\hline",
    ]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)
