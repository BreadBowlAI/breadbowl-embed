"""Evaluation data: BEIR-layout corpus/queries/qrels, TREC qrels and TREC run files.

Accepted inputs (plain or .gz):

- corpus: JSONL rows {"_id" (or "id"), "title" (optional), "text"}; a TSV of "id<TAB>text" or
  "id<TAB>title<TAB>text" rows.
- queries: JSONL rows {"_id" (or "id"), "text"} (or "query" instead of "text"); a TSV of
  "id<TAB>text" rows.
- qrels: BEIR TSV with a "query-id corpus-id score" header, or TREC qrels "qid 0 docid relevance"
  (whitespace separated, as distributed for TREC DL and MS MARCO).
- runs: TREC format "qid Q0 docid rank score tag".

Everything is returned as plain dicts, so callers can equally pass in-memory mappings.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil

DOCUMENT_FORMATS = ("title_text", "text")


def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def _is_jsonl(path: Path) -> bool:
    suffixes = [s for s in path.suffixes if s != ".gz"]
    return bool(suffixes) and suffixes[-1] in {".jsonl", ".json"}


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open(path) as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg}).") from None
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object.")
                yield row


def _row_id(row: dict, path, number) -> str:
    value = row.get("_id", row.get("id"))
    if value is None or str(value) == "":
        raise ValueError(f"{path}:{number}: missing '_id'.")
    return str(value)


def load_corpus(path: str | Path) -> dict[str, dict[str, str]]:
    """document id -> {"title", "text"}, in file order. Duplicate ids are an error."""
    path = Path(path)
    corpus: dict[str, dict[str, str]] = {}

    def put(doc_id, title, text, number):
        if doc_id in corpus:
            raise ValueError(f"{path}:{number}: duplicate document id {doc_id!r}.")
        if not isinstance(text, str) or not isinstance(title, str):
            raise ValueError(f"{path}:{number}: 'title' and 'text' must be strings.")
        corpus[doc_id] = {"title": title, "text": text}

    if _is_jsonl(path):
        for number, row in enumerate(read_jsonl(path), 1):
            put(_row_id(row, path, number), row.get("title") or "", row.get("text"), number)
    else:
        with _open(path) as handle:
            for number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                if number == 1 and fields[0].lower() in {"id", "_id", "doc_id", "docid", "corpus-id"}:
                    continue
                if not line.strip():
                    continue
                if len(fields) == 2:
                    put(fields[0], "", fields[1], number)
                elif len(fields) == 3:
                    put(fields[0], fields[1], fields[2], number)
                else:
                    raise ValueError(f"{path}:{number}: expected id<TAB>text or id<TAB>title<TAB>text.")
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    """query id -> query text."""
    path = Path(path)
    queries: dict[str, str] = {}
    if _is_jsonl(path):
        for number, row in enumerate(read_jsonl(path), 1):
            text = row.get("text", row.get("query"))
            if not isinstance(text, str):
                raise ValueError(f"{path}:{number}: expected a string 'text'.")
            qid = _row_id(row, path, number)
            if qid in queries:
                raise ValueError(f"{path}:{number}: duplicate query id {qid!r}.")
            queries[qid] = text
    else:
        with _open(path) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                qid, sep, text = line.rstrip("\n").partition("\t")
                if not sep:
                    raise ValueError(f"{path}:{number}: expected id<TAB>text.")
                if qid in queries:
                    raise ValueError(f"{path}:{number}: duplicate query id {qid!r}.")
                queries[qid] = text
    return queries


def _relevance(value: str, path, number) -> int:
    try:
        number_value = float(value)
    except ValueError:
        raise ValueError(f"{path}:{number}: relevance {value!r} is not a number.") from None
    if not math.isfinite(number_value) or number_value != int(number_value):
        raise ValueError(f"{path}:{number}: relevance {value!r} is not an integer.")
    return int(number_value)


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """query id -> {document id -> relevance}. BEIR TSV (with header) or TREC qrels."""
    qrels: dict[str, dict[str, int]] = {}
    with _open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            if number == 1 and fields[0].strip().lower() in {"query-id", "query_id", "qid"}:
                continue
            fields = [f.strip() for f in fields]
            if len(fields) == 3:
                qid, doc, rel = fields
            elif len(fields) == 4:
                qid, _, doc, rel = fields
            else:
                raise ValueError(f"{path}:{number}: expected 'query-id corpus-id score' or 'qid 0 docid rel'.")
            judgments = qrels.setdefault(qid, {})
            if doc in judgments:
                raise ValueError(f"{path}:{number}: duplicate judgment for ({qid}, {doc}).")
            judgments[doc] = _relevance(rel, path, number)
    return qrels


def read_run(path: str | Path) -> dict[str, dict[str, float]]:
    """A TREC run file as query id -> {document id -> score}. The rank column is not used."""
    run: dict[str, dict[str, float]] = {}
    with _open(path) as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{number}: expected 'qid Q0 docid rank score tag'.")
            qid, _, doc, _, score, _ = fields
            try:
                value = float(score)
            except ValueError:
                raise ValueError(f"{path}:{number}: score {score!r} is not a number.") from None
            if math.isnan(value):
                raise ValueError(f"{path}:{number}: score is NaN.")
            documents = run.setdefault(qid, {})
            if doc in documents:
                raise ValueError(f"{path}:{number}: document {doc!r} listed twice for query {qid!r}.")
            documents[doc] = value
    return run


def write_run(path: str | Path, run: Mapping[str, Sequence[tuple[str, float]]], *, tag: str = "breadbowl") -> None:
    """Write ranked (document id, score) lists in TREC format, in the order given.

    Scores are written with full precision (repr), so reading the file back and ranking by
    descending score with ties by ascending id reproduces the order. trec_eval orders tied scores by
    descending id instead, which matters only for exactly equal scores.
    """
    if not tag or any(c.isspace() for c in tag):
        raise ValueError("The run tag must be one word.")
    with open(path, "w", encoding="utf-8") as handle:
        for qid in sorted(run):
            for position, (doc, score) in enumerate(run[qid], 1):
                if any(c.isspace() for c in str(qid) + str(doc)):
                    raise ValueError("TREC run files cannot hold ids with whitespace.")
                handle.write(f"{qid} Q0 {doc} {position} {float(score)!r} {tag}\n")


def document_text(document: Mapping[str, str] | str, document_format: str = "title_text") -> str:
    """The text encoded for one corpus entry.

    "title_text": (title + " " + text).strip() when the title is non-empty, else text.strip(), the
    rule MTEB uses for retrieval corpora; "text": the text alone, stripped.
    """
    if document_format not in DOCUMENT_FORMATS:
        raise ValueError(f"document_format must be one of {DOCUMENT_FORMATS}.")
    if isinstance(document, str):
        return document.strip()
    title, text = document.get("title") or "", document.get("text") or ""
    if document_format == "title_text" and title:
        return (title + " " + text).strip()
    return text.strip()


@dataclass
class Dataset:
    """corpus, queries and qrels of one split, plus where they came from."""
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    source: dict

    def describe(self) -> dict:
        return {**self.source, "documents": len(self.corpus), "queries": len(self.queries),
                "judged_queries": len(self.qrels), "queries_sha256": fingerprint_queries(self.queries),
                "qrels_sha256": fingerprint_qrels(self.qrels)}


def _first(directory: Path, names: Sequence[str]) -> Path:
    for name in names:
        for candidate in (directory / name, directory / (name + ".gz")):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"None of {list(names)} in {directory}.")


def load_beir(directory: str | Path, split: str = "test", *, corpus: bool = True) -> Dataset:
    """A BEIR-layout directory: corpus.jsonl, queries.jsonl and qrels/<split>.tsv."""
    root = Path(directory)
    qrels_path = _first(root / "qrels", [f"{split}.tsv", f"{split}.txt"])
    queries_path = _first(root, ["queries.jsonl", "queries.tsv"])
    return Dataset(load_beir_corpus(root) if corpus else {}, load_queries(queries_path), load_qrels(qrels_path),
                   {"name": root.name, "split": split})


def load_beir_corpus(directory: str | Path) -> dict[str, dict[str, str]]:
    """corpus.jsonl (or corpus.tsv, optionally .gz) of a BEIR-layout directory."""
    return load_corpus(_first(Path(directory), ["corpus.jsonl", "corpus.tsv"]))


def fingerprint_queries(queries: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for qid in sorted(queries):
        digest.update(json.dumps([qid, queries[qid]], ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def fingerprint_qrels(qrels: Mapping[str, Mapping[str, int]]) -> str:
    digest = hashlib.sha256()
    for qid in sorted(qrels):
        for doc in sorted(qrels[qid]):
            digest.update(json.dumps([qid, doc, qrels[qid][doc]]).encode() + b"\n")
    return digest.hexdigest()


def corpus_items(corpus: Mapping[str, Mapping[str, str] | str] | Iterable[tuple[str, str]],
                 document_format: str = "title_text") -> list[tuple[str, str]]:
    """(id, text) pairs in the given order, with the document format applied."""
    pairs = corpus.items() if isinstance(corpus, Mapping) else corpus
    items = [(str(doc_id), document_text(doc, document_format)) for doc_id, doc in pairs]
    if len({doc_id for doc_id, _ in items}) != len(items):
        raise ValueError("Document ids must be unique.")
    return items


def disk_free(path: str | Path) -> int:
    """Free bytes on the file system that holds `path` (or its nearest existing parent)."""
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free
