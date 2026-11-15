"""Resumable corpus encoding into the library's on-disk `Index` format.

Encoding is the slow part of an evaluation on CPU, so a build directory is written in parts that
survive interruption:

    <directory>/build.json           what is being built: model identity, dtype, batch size,
                                     document format, corpus fingerprint, chunk size
    <directory>/parts/part-00000/    an ordinary Index directory (float32) for documents 0..chunk-1
    <directory>/parts/part-00001/    ... written to a hidden temporary name, then renamed
    <directory>/timings/part-*.json  encoding seconds, threads and torch version of each part
    <directory>/index-<storage>/     the merged Index, written once every part exists

Rerunning the same call skips finished parts, so an interrupted build continues where it stopped.
A build directory refuses to continue with a different model, dtype, batch size, document
format, chunk size, library version or corpus. Parts are kept in float32, so the merged index
with any storage type equals encoding the whole corpus at once and calling `model.index(...)`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .._version import __version__
from ..config import canonical_json
from ..embeddings import Embeddings
from ..retrieval import STORAGE_TYPES, Index, resolve_profile
from .data import corpus_items, disk_free

BUILD_FORMAT = "breadbowl_evaluation_build_v1"
PART_STORAGE = "float32"
# Settings that decide the encodings; a resumed build must match all of them.
IDENTITY_KEYS = ("format", "library_version", "representation_id", "compute_dtype", "batch_size",
                 "document_format", "documents", "corpus_sha256", "chunk_size")


def corpus_fingerprint(items: Iterable[tuple[str, str]]) -> str:
    """sha256 over the ordered (id, encoded text) pairs."""
    digest = hashlib.sha256()
    for doc_id, text in items:
        digest.update(json.dumps([doc_id, text], ensure_ascii=False).encode() + b"\n")
    return digest.hexdigest()


def _bytes_per_document(model) -> int:
    architecture = model.config["architecture"]
    slots = architecture["slot_extractor"]["num_slots"]
    return slots * (architecture["heads"]["d_k"] + architecture["heads"]["d_v"]) * 4


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_build(directory: str | Path) -> dict:
    """build.json of a build directory, with the parts' encoding timings summed."""
    root = Path(directory)
    info = json.loads((root / "build.json").read_text())
    if info.get("format") != BUILD_FORMAT:
        raise ValueError(f"{root} is not an evaluation build directory.")
    seconds = documents = 0
    for timing in sorted((root / "timings").glob("part-*.json")):
        entry = json.loads(timing.read_text())
        seconds += entry["seconds"]
        documents += entry["documents"]
    info["encoding_timing"] = {"documents": documents, "seconds": seconds,
                               "documents_per_second": documents / seconds if seconds else None}
    return info


def build_index(model, corpus: Mapping | Iterable[tuple[str, str]], directory: str | Path, *,
                document_format: str = "title_text", storage: str = "float32", profile=None,
                chunk_size: int = 256, batch_size: int = 1, shard_size: int = 16384, keep_parts: bool = True,
                check_disk: bool = True, progress: Callable[[int, int, int, float], None] | None = None) -> Index:
    """Encode `corpus` into `directory` (resumable) and return the merged Index.

    corpus: document id -> {"title", "text"} (BEIR) or text, or (id, text) pairs; encoded in the
    given order. document_format: "title_text" ((title + " " + text).strip(), MTEB's rule) or
    "text". storage/profile: of the merged index (parts are always float32). chunk_size: documents
    per resumable part. batch_size: texts per forward pass (1, the library default, keeps every
    encoding independent of its neighbours). shard_size: documents per shard of the merged index.
    progress(done, total, encoded_this_call, seconds_this_call) is called after every part.
    """
    if storage not in STORAGE_TYPES:
        raise ValueError(f"storage must be one of {STORAGE_TYPES}.")
    for name, value in (("chunk_size", chunk_size), ("batch_size", batch_size), ("shard_size", shard_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    items = corpus_items(corpus, document_format)
    if not items:
        raise ValueError("The corpus is empty.")
    for doc_id, text in items:
        if not text:
            raise ValueError(f"Document {doc_id!r} has no text; remove it or give it text.")
    root = Path(directory)
    plan = {"format": BUILD_FORMAT, "library_version": __version__, "representation_id": model.representation_id,
            "release_id": model.release_id, "compute_dtype": model.compute_dtype, "batch_size": batch_size,
            "document_format": document_format, "documents": len(items), "corpus_sha256": corpus_fingerprint(items),
            "chunk_size": chunk_size, "part_storage": PART_STORAGE,
            "max_document_length": model.config["preprocessing"]["max_document_length"]}
    parts_dir = root / "parts"
    if (root / "build.json").is_file():
        existing = json.loads((root / "build.json").read_text())
        changed = [key for key in IDENTITY_KEYS if existing.get(key) != plan[key]]
        if changed:
            raise ValueError(f"{root} holds a build with different {', '.join(changed)}; resume it with the same "
                             "settings or use a new directory.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} exists and is not an evaluation build directory.")
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "build.json", plan)
    parts_dir.mkdir(exist_ok=True)
    count = (len(items) + chunk_size - 1) // chunk_size
    names = [f"part-{number:05d}" for number in range(count)]
    target = root / f"index-{storage}"
    finished = (target / "index.json").is_file()
    todo = [] if finished else [n for n, name in enumerate(names) if not (parts_dir / name / "index.json").is_file()]
    if check_disk and not finished:
        remaining = sum(len(items[n * chunk_size:(n + 1) * chunk_size]) for n in todo)
        per_document = _bytes_per_document(model)
        needed = remaining * per_document + len(items) * per_document  # parts, then the merged index
        if disk_free(root) < needed * 1.05:
            raise OSError(f"About {needed / 2**30:.1f} GiB are needed under {root} and "
                          f"{disk_free(root) / 2**30:.1f} GiB are free.")
    started = time.monotonic()
    done = len(items) - sum(len(items[n * chunk_size:(n + 1) * chunk_size]) for n in todo)
    encoded_now = 0
    for number in todo:
        final = parts_dir / names[number]
        if final.exists():
            raise ValueError(f"{final} exists without index.json; it was not written by this tool. Remove it.")
        chunk = items[number * chunk_size:(number + 1) * chunk_size]
        temporary = parts_dir / f".{names[number]}.partial"
        if temporary.exists():
            shutil.rmtree(temporary)  # an interrupted write of this tool
        part_start = time.monotonic()
        encoded = model.encode_documents([text for _, text in chunk], batch_size=batch_size)
        seconds = time.monotonic() - part_start
        model.index(encoded, ids=[doc_id for doc_id, _ in chunk], storage=PART_STORAGE).save(temporary)
        os.rename(temporary, final)
        (root / "timings").mkdir(exist_ok=True)
        _write_json(root / "timings" / f"{names[number]}.json", _timing(len(chunk), seconds))
        done += len(chunk)
        encoded_now += len(chunk)
        if progress is not None:
            progress(done, len(items), encoded_now, time.monotonic() - started)
    if finished:
        index = model.load_index(target)
    else:
        index = _merge(model, parts_dir, names, items, chunk_size, storage, profile, shard_size)
        temporary = root / f".index-{storage}.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        index.save(temporary)
        os.rename(temporary, target)
    if index.ids != [doc_id for doc_id, _ in items]:
        raise ValueError(f"{target} does not hold the documents of this build.")
    if profile is not None:
        index.profile = resolve_profile(profile)
    if not keep_parts and parts_dir.exists():
        shutil.rmtree(parts_dir)  # the merged index stays; another storage type then needs a new encoding
    return index


def _timing(documents: int, seconds: float) -> dict:
    import torch
    return {"documents": documents, "seconds": seconds, "threads": torch.get_num_threads(),
            "torch": torch.__version__}


def _merge(model, parts_dir: Path, names: list[str], items, chunk_size: int, storage: str, profile,
           shard_size: int) -> Index:
    index = None
    buffered: list[Embeddings] = []
    buffered_ids: list[str] = []

    def flush():
        nonlocal index
        if not buffered:
            return
        embeddings = Embeddings.concat(buffered)
        if index is None:
            index = Index(embeddings, list(buffered_ids), scoring=model.scoring, storage=storage, profile=profile)
        else:
            index.add(embeddings, list(buffered_ids))
        buffered.clear()
        buffered_ids.clear()

    for number, name in enumerate(names):
        part = Index.load(parts_dir / name, representation_id=model.representation_id)
        expected = [doc_id for doc_id, _ in items[number * chunk_size:(number + 1) * chunk_size]]
        if part.ids != expected or part.storage != PART_STORAGE:
            raise ValueError(f"{parts_dir / name} does not hold the expected documents.")
        rows = list(range(len(part)))
        # Stored float32 vectors, read back unchanged (not re-normalised).
        buffered.append(Embeddings(part.routing_vectors(rows), part.value_vectors(rows), part.representation_id,
                                   "document"))
        buffered_ids.extend(part.ids)
        if len(buffered_ids) >= shard_size:
            flush()
    flush()
    return index


def describe_index_source(index_or_directory) -> tuple[Index | None, dict]:
    """What the evaluation records about where the document vectors came from."""
    if isinstance(index_or_directory, Index):
        return index_or_directory, {"source": "in-memory index", "documents": len(index_or_directory),
                                    "storage": index_or_directory.storage,
                                    "ids_sha256": hashlib.sha256(canonical_json(index_or_directory.ids).encode())
                                    .hexdigest()}
    root = Path(index_or_directory)
    if (root / "build.json").is_file():
        info = read_build(root)
        merged = sorted(root.glob("index-*/index.json"))
        return None, {"source": "evaluation build", **{k: info[k] for k in IDENTITY_KEYS if k in info},
                      "encoding_timing": info["encoding_timing"], "merged_indexes": [p.parent.name for p in merged]}
    manifest = json.loads((root / "index.json").read_text())
    return None, {"source": "index directory", "documents": manifest.get("count"), "storage": manifest.get("storage"),
                  "library_version": manifest.get("library_version"),
                  "corpus_encoding": "not recorded by the index format (batch size, document format unknown)"}


def load_for_evaluation(model, index_or_directory, *, storage: str | None = None,
                        mmap: bool = False) -> tuple[Index, dict]:
    """An Index plus a description, from an Index, an Index directory or a build directory."""
    index, description = describe_index_source(index_or_directory)
    if index is not None:
        if index.representation_id != model.representation_id:
            raise ValueError("The index was built by a different model representation (model or dtype).")
        return index, description
    root = Path(index_or_directory)
    if description["source"] == "evaluation build":
        candidates = [root / f"index-{storage}"] if storage else sorted(root.glob("index-*"))
        candidates = [c for c in candidates if (c / "index.json").is_file()]
        if len(candidates) != 1:
            raise ValueError(f"{root}: expected one merged index, found {[c.name for c in candidates]}; "
                             "finish the build or pass storage=.")
        root = candidates[0]
    index = model.load_index(root, mmap=mmap)
    description["storage"] = index.storage
    description["documents"] = len(index)
    description["ids_sha256"] = hashlib.sha256(canonical_json(index.ids).encode()).hexdigest()
    return index, description
