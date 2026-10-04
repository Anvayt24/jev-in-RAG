"""The document downloader, exercised against a local HTTP server and synthetic PDFs."""

import functools
import http.server
import json
import threading
from pathlib import Path

import pymupdf
import pytest

from jevbench import download_docs
from jevbench.download_docs import SOURCES, Source, cut_pages, download_all, fetch, page_count


def make_pdf(path: Path, pages: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        for i in range(1, pages + 1):
            doc.new_page().insert_text((72, 72), f"page-{i}")
        doc.save(path)
    return path


def page_texts(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text().strip() for page in doc]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def site(tmp_path):
    """Serve a few files from a temporary folder on localhost; yields (base_url, folder)."""
    root = tmp_path / "www"
    root.mkdir()
    make_pdf(root / "paper.pdf", pages=5)
    make_pdf(root / "report.pdf", pages=30)
    (root / "blocked.pdf").write_text("<html>access denied</html>")
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", root
    server.shutdown()
    server.server_close()


def test_fetch_saves_a_pdf_and_leaves_no_partial_file(site, tmp_path):
    base, _ = site
    dest = tmp_path / "out" / "paper.pdf"
    dest.parent.mkdir()
    fetch(f"{base}/paper.pdf", dest)
    assert page_count(dest) == 5
    assert list(dest.parent.iterdir()) == [dest]


def test_fetch_rejects_a_response_that_is_not_a_pdf(site, tmp_path):
    base, _ = site
    dest = tmp_path / "blocked.pdf"
    with pytest.raises(ValueError, match="did not return a PDF"):
        fetch(f"{base}/blocked.pdf", dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_cut_pages_keeps_the_requested_ranges_in_order(tmp_path):
    src = make_pdf(tmp_path / "full.pdf", pages=12)
    dest = tmp_path / "cut.pdf"
    assert cut_pages(src, dest, [(2, 3), (7, 7), (11, 12)]) == 5
    assert page_texts(dest) == ["page-2", "page-3", "page-7", "page-11", "page-12"]


def test_download_all_saves_cuts_keeps_and_warns(site, tmp_path):
    base, _ = site
    sources = (
        Source("paper.pdf", f"{base}/paper.pdf", pages=5),
        Source("revised.pdf", f"{base}/paper.pdf", pages=7),  # publisher revised the document
        Source("excerpt.pdf", f"{base}/report.pdf", pages=4, keep_pages=((2, 3), (10, 11))),
    )
    out = tmp_path / "raw"
    lines = download_all(out, sources=sources)
    assert lines[0] == "saved  paper.pdf (5 pages)"
    assert "WARNING: expected 7 pages, got 5" in lines[1]
    assert lines[2] == "saved  excerpt.pdf (4 pages)"
    assert page_texts(out / "excerpt.pdf") == ["page-2", "page-3", "page-10", "page-11"]

    again = download_all(out, sources=sources)
    assert all(line.startswith("kept") for line in again)

    (out / "paper.pdf").write_bytes(b"stale")
    download_all(out, force=True, sources=sources)
    assert page_count(out / "paper.pdf") == 5


def test_main_reports_progress_and_next_steps(site, tmp_path, monkeypatch, capsys):
    base, _ = site
    monkeypatch.setattr(download_docs, "SOURCES", (Source("paper.pdf", f"{base}/paper.pdf"),))
    monkeypatch.setattr(
        download_docs,
        "download_all",
        functools.partial(download_all, sources=download_docs.SOURCES),
    )
    assert download_docs.main(["--out", str(tmp_path / "raw")]) == 0
    out = capsys.readouterr().out
    assert "saved  paper.pdf (5 pages)" in out
    assert "jevbench.ingest" in out


def test_berkshire_excerpt_ranges_add_up_to_the_declared_page_count():
    berkshire = next(s for s in SOURCES if s.filename.startswith("berkshire"))
    assert sum(last - first + 1 for first, last in berkshire.keep_pages) == berkshire.pages == 33


def test_every_document_the_eval_set_cites_can_be_downloaded():
    questions = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    cited = set()
    for line in questions.read_text(encoding="utf-8").splitlines():
        for gold in json.loads(line)["gold"]:
            cited.add(gold["doc_id"])
    available = {Path(s.filename).stem for s in SOURCES}
    assert cited == available
