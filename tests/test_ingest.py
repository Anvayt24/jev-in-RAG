"""Chunking and tokenisation, tested with a word-count stand-in for the real tokenizer."""

from jevbench import config
from jevbench.ingest import bm25_tokens, chunk_document, split_text


def words(text: str) -> int:
    return len(text.split())


def test_short_text_is_a_single_chunk():
    assert split_text("one two three", words, size=10) == ["one two three"]


def test_chunks_respect_the_size_limit_and_keep_all_words_in_order():
    paragraphs = [" ".join(f"w{p}_{i}" for i in range(40)) for p in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, words, size=50)
    assert all(words(c) <= 50 for c in chunks)
    assert all(c.strip() for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_one_oversized_paragraph_is_split_at_sentence_then_word_boundaries():
    sentence = "alpha beta gamma delta epsilon."
    text = " ".join([sentence] * 30)  # one 150-word paragraph
    chunks = split_text(text, words, size=40)
    assert len(chunks) > 1
    assert all(words(c) <= 40 for c in chunks)
    # splitting on ". " consumes the separator, so a chunk ending at a sentence boundary loses its
    # final period (documented in split_text); every word is otherwise preserved, in order
    assert " ".join(chunks).replace(".", "").split() == text.replace(".", "").split()
    assert chunks[0].endswith("epsilon")


def test_a_single_unbreakable_run_of_words_is_still_split():
    text = " ".join(f"tok{i}" for i in range(100))
    chunks = split_text(text, words, size=30)
    assert all(words(c) <= 30 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chunk_ids_and_pages_are_stable():
    pages = [{"page": 3, "text": "short page"}, {"page": 7, "text": "another page"}]
    chunks = chunk_document("doc", pages, words)
    assert [c["chunk_id"] for c in chunks] == ["doc:p3:c0", "doc:p7:c0"]
    assert [c["page"] for c in chunks] == [3, 7]
    assert all(c["doc_id"] == "doc" for c in chunks)


def test_long_page_gets_overlapping_chunks_that_never_exceed_the_window():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_document("doc", [{"page": 1, "text": text}], words)
    assert len(chunks) >= 3
    assert all(words(c["text"]) <= config.CHUNK_TOKENS for c in chunks)
    # each chunk starts with the last CHUNK_OVERLAP_TOKENS words of the one before it
    n = config.CHUNK_OVERLAP_TOKENS
    assert chunks[1]["text"].split()[:n] == chunks[0]["text"].split()[-n:]


def test_bm25_tokens_lowercase_and_keep_numbers_and_hyphenated_terms():
    tokens = bm25_tokens("Front-line staff earned 3.5% more in FY2023, up 1,200 units.")
    assert "front-line" in tokens
    assert "3.5" in tokens
    assert "fy2023" in tokens
    assert all(t == t.lower() for t in tokens)
