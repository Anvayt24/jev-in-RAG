"""Parse PDFs -> markdown pages -> chunks -> dense (FAISS) + sparse (BM25) indexes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import config


def parse_pdf(path: Path) -> list[dict]:
    """Return [{'page': int (1-based), 'text': str}] using pymupdf4llm markdown."""
    cache = config.PARSED_DIR / f"{path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    out = []
    for i, p in enumerate(pages):
        text = p["text"].strip()
        if text:
            out.append({"page": i + 1, "text": text})
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def split_text(text: str, tok_len, size: int) -> list[str]:
    """Recursive splitter: paragraphs -> lines -> sentences -> words, merged up to `size` tokens.

    Known quirk, kept so published results stay reproducible: when a split lands on a ". "
    boundary the separator is consumed, so that chunk ends without its final period.
    """

    def rec(t: str, seps: list[str]) -> list[str]:
        if tok_len(t) <= size:
            return [t]
        if not seps:
            words = t.split(" ")
            mid = len(words) // 2
            return rec(" ".join(words[:mid]), []) + rec(" ".join(words[mid:]), [])
        sep, rest = seps[0], seps[1:]
        out: list[str] = []
        cur = ""
        for part in t.split(sep):
            cand = f"{cur}{sep}{part}" if cur else part
            if tok_len(cand) <= size:
                cur = cand
                continue
            if cur:
                out.append(cur)
            if tok_len(part) > size:
                out.extend(rec(part, rest))
                cur = ""
            else:
                cur = part
        if cur:
            out.append(cur)
        return out

    return [c for c in rec(text, ["\n\n", "\n", ". ", " "]) if c.strip()]


def _tail_tokens(text: str, tok_len, n_tokens: int) -> str:
    """Last whitespace-delimited words of `text` totalling at most ~n_tokens tokens."""
    words = text.split()
    tail: list[str] = []
    for w in reversed(words):
        if tok_len(" ".join([w, *tail])) > n_tokens:
            break
        tail.insert(0, w)
    return " ".join(tail)


def chunk_document(doc_id: str, pages: list[dict], tok_len) -> list[dict]:
    chunks: list[dict] = []
    for pg in pages:
        # leave room for the token overlap prepended below
        parts = split_text(pg["text"], tok_len, config.CHUNK_TOKENS - config.CHUNK_OVERLAP_TOKENS)
        prev_tail = ""
        for i, part in enumerate(parts):
            text = f"{prev_tail} {part}".strip() if prev_tail else part
            if tok_len(text) > config.CHUNK_TOKENS:  # never exceed the embedder's window
                text = part
            chunks.append(
                {
                    "chunk_id": f"{doc_id}:p{pg['page']}:c{i}",
                    "doc_id": doc_id,
                    "page": pg["page"],
                    "text": text,
                }
            )
            prev_tail = _tail_tokens(part, tok_len, config.CHUNK_OVERLAP_TOKENS)
    return chunks


def build() -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    tok = _get_tokenizer()
    tok_len = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

    all_chunks: list[dict] = []
    for pdf in sorted(config.RAW_DIR.glob("*.pdf")):
        pages = parse_pdf(pdf)
        ch = chunk_document(pdf.stem, pages, tok_len)
        print(f"{pdf.stem:36s} pages={len(pages):3d} chunks={len(ch):4d}")
        all_chunks.extend(ch)

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lens = [tok_len(c["text"]) for c in all_chunks]
    print(
        f"total chunks: {len(all_chunks)}  "
        f"tokens/chunk: median={int(np.median(lens))} max={max(lens)}",
        flush=True,
    )

    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    emb = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    print("index built:", index.ntotal, "vectors")


def bm25_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())


if __name__ == "__main__":
    build()
