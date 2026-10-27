from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factcheck.config import get_settings
from factcheck.documents import load_document
from factcheck.factory import build_system
from factcheck.formatting import format_audit_report, label_text
from factcheck.observability import flush


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a document for checkable scientific claims.")
    parser.add_argument("path", help="PDF, TXT or MD file")
    parser.add_argument("--max-claims", type=int, default=None, help="maximum claims to check")
    parser.add_argument("--out", default=None, help="where to write the Markdown report")
    args = parser.parse_args()

    settings = get_settings()
    text = load_document(args.path)
    _, auditor = build_system(settings)

    def show(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}", flush=True)

    report = auditor.audit(text, args.max_claims, progress=show)
    markdown = format_audit_report(report)
    out = Path(args.out) if args.out else settings.reports_dir / f"audit_{Path(args.path).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    print()
    for item in report.items:
        print(f"{item.index:>2}. {label_text(item.label):<16} {item.claim}")
    print(f"\nReport written to {out}")
    flush()


if __name__ == "__main__":
    main()
