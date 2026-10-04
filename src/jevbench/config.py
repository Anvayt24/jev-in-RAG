"""Shared paths and constants for the benchmark."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
CACHE_DIR = ROOT / "cache"
PARSED_DIR = CACHE_DIR / "parsed"
INDEX_DIR = CACHE_DIR / "index"
CAND_DIR = CACHE_DIR / "candidates"
JEV_RAW_DIR = CACHE_DIR / "jev_raw"
RESULTS_DIR = ROOT / "results"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_TOKENS = 480  # bge-small max is 512 incl. special tokens
CHUNK_OVERLAP_TOKENS = 50

RRF_K = 60
RETRIEVE_PER_RETRIEVER = 50
N_CANDIDATES = 30

JEV_MODEL = "typesafe/jev-1.13"
JEV_BASE_URL = "https://openrouter.ai/api"
JEV_PRICE_PER_INPUT_TOKEN = 0.042e-6  # USD, matches usage.cost seen via OpenRouter
