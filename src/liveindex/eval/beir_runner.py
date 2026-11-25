"""Load BEIR datasets (SciFact, NFCorpus) into a local cache folder.

The official BEIR host (public.ukp.informatik.tu-darmstadt.de) was unreachable from this
machine, so files are fetched from the Hugging Face mirrors (BeIR/<name>, BeIR/<name>-qrels),
written in the standard BEIR folder layout, and then read with the `beir` GenericDataLoader.
"""
from __future__ import annotations

import json
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "data" / "beir"
SPLITS = {"scifact": "test", "nfcorpus": "test"}


def _materialize(name: str, folder: Path) -> None:
    import pandas as pd

    folder.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(
        hf_hub_download(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
    )
    queries = pd.read_parquet(
        hf_hub_download(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset")
    )
    with open(folder / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"_id": str(r._1), "title": r.title or "", "text": r.text or ""}) + "\n" for r in corpus.itertuples())
    with open(folder / "queries.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"_id": str(r._1), "text": r.text}) + "\n" for r in queries.itertuples())
    (folder / "qrels").mkdir(exist_ok=True)
    for split in ("test",):
        p = hf_hub_download(f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset")
        (folder / "qrels" / f"{split}.tsv").write_bytes(Path(p).read_bytes())


def load(name: str, cache_dir: Path | None = None, split: str | None = None):
    """Return (corpus, queries, qrels). corpus: id -> {title, text}; qrels: qid -> {did: rel}."""
    split = split or SPLITS[name]
    folder = (cache_dir or DEFAULT_CACHE) / name
    if not (folder / "corpus.jsonl").exists():
        _materialize(name, folder)
    return GenericDataLoader(data_folder=str(folder)).load(split=split)


def doc_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title}. {text}" if title else text
