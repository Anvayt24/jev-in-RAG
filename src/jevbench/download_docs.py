"""Download the benchmark's source documents into data/raw/.

  uv run python -m jevbench.download_docs

Six documents are saved as published; the Berkshire Hathaway annual report is cut down to the
33 pages the eval set draws on. Files that already exist are kept unless --force is given.

Afterwards run `python -m jevbench.ingest` and `python -m jevbench.build_eval`. If a publisher has
revised a document, build_eval reports every gold quote that no longer matches.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config

USER_AGENT = "Mozilla/5.0 (compatible; jevbench/0.1; +https://github.com/Anvayt24/jev)"


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    pages: int | None = None  # expected page count of the saved file (a sanity check, not a gate)
    keep_pages: tuple[tuple[int, int], ...] | None = None  # 1-based inclusive ranges to keep


SOURCES: tuple[Source, ...] = (
    Source("rag_lewis_2020.pdf", "https://arxiv.org/pdf/2005.11401", pages=19),
    Source("lost_in_the_middle_2023.pdf", "https://arxiv.org/pdf/2307.03172", pages=18),
    Source("self_rag_2023.pdf", "https://arxiv.org/pdf/2310.11511", pages=30),
    Source(
        "nist_ai_rmf_1_0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        pages=48,
    ),
    Source(
        "nist_csf_2_0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        pages=32,
    ),
    Source(
        "raspberry_pi_pico_datasheet.pdf",
        "https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf",
        pages=31,
    ),
    Source(
        "berkshire_2023_excerpt.pdf",
        "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
        pages=33,
        keep_pages=((5, 19), (59, 63), (66, 67), (81, 82), (94, 98), (136, 139)),
    ),
)


def fetch(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Download `url` to `dest`, refusing anything that is not a PDF. The write is atomic."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = dest.with_name(dest.name + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        with partial.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError(f"{url} did not return a PDF")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


def page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as doc:
        return len(doc)


def cut_pages(src: Path, dest: Path, ranges: Sequence[tuple[int, int]]) -> int:
    """Write the given 1-based inclusive page ranges of `src` to `dest`; returns the page count."""
    import pymupdf

    with pymupdf.open(src) as doc, pymupdf.open() as out:
        for first, last in ranges:
            out.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
        out.save(dest)
        return len(out)


def download_all(
    out_dir: Path,
    force: bool = False,
    sources: Sequence[Source] = SOURCES,
    fetch_fn: Callable[[str, Path], None] = fetch,
) -> list[str]:
    """Fetch every source into `out_dir` and return one status line per document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for source in sources:
        dest = out_dir / source.filename
        if dest.exists() and not force:
            status.append(f"kept   {source.filename} (already present)")
            continue
        if source.keep_pages:
            with tempfile.TemporaryDirectory() as tmp:
                full = Path(tmp) / "full.pdf"
                fetch_fn(source.url, full)
                pages = cut_pages(full, dest, source.keep_pages)
        else:
            fetch_fn(source.url, dest)
            pages = page_count(dest)
        note = ""
        if source.pages is not None and source.pages != pages:
            note = (
                f"  WARNING: expected {source.pages} pages, got {pages}; it may have been revised"
            )
        status.append(f"saved  {source.filename} ({pages} pages){note}")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=config.RAW_DIR, help="target folder")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args(argv)
    for line in download_all(args.out, force=args.force):
        print(line)
    print("\nNext: python -m jevbench.ingest && python -m jevbench.build_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
