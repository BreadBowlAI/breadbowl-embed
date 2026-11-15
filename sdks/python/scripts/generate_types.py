#!/usr/bin/env python3
"""Generate ``src/breadbowl/_generated.py`` from the public OpenAPI contract.

    python scripts/generate_types.py          # write
    python scripts/generate_types.py --check  # exit 1 if the file is stale

Generates what can be derived mechanically: ``Literal`` aliases for every enum, request
``TypedDict``s (for callers who build payloads themselves), ``TypedDict``s for the typed error
``details`` (``x-error-details``), the operation table (method, path, scopes, whether
``Idempotency-Key`` is accepted, ``x-error-codes``), the error codes per HTTP status and the
default server URL. The response dataclasses in ``_models.py`` are handwritten for ergonomics;
``tests/test_contract.py`` proves they agree with the same schemas.

Needs PyYAML (``pip install -e ".[dev]"``).
"""

from __future__ import annotations

import hashlib
import keyword
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
SPEC_PATH = REPO_ROOT / "sdks" / "openapi" / "bb-embed.v1.yaml"
OUT_PATH = HERE.parent / "src" / "breadbowl" / "_generated.py"

METHODS = ("get", "post", "put", "patch", "delete")


def load_spec() -> Dict[str, Any]:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def deref(spec: Dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        node_ref = node["$ref"]
        cur: Any = spec
        for part in node_ref.lstrip("#/").split("/"):
            cur = cur[part]
        node = cur
    return node


def pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\-\s]+", name) if p)


class Gen:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.spec = spec
        self.schemas: Dict[str, Any] = spec["components"]["schemas"]
        self.literals: Dict[str, List[Any]] = {}
        self.typeddicts: List[str] = []
        self.aliases: List[str] = []

    def collect_aliases(self) -> None:
        """Named scalar, union and map schemas (``DocId``, ``MetadataValue``, ``Metadata``)."""
        for name, schema in self.schemas.items():
            if name in self.literals or "enum" in schema:
                continue
            typ = schema.get("type")
            scalar = typ in ("string", "integer", "number", "boolean") or (
                isinstance(typ, list) and "object" not in typ and "array" not in typ
            )
            mapping = (
                typ == "object"
                and isinstance(schema.get("additionalProperties"), dict)
                and not schema.get("properties")
            )
            if scalar or mapping:
                self.aliases.append(name)

    def alias_lines(self) -> List[str]:
        ordered: List[str] = []

        def visit(n: str) -> None:
            if n in ordered:
                return
            for ref in re.findall(r"#/components/schemas/(\w+)", repr(self.schemas[n])):
                if ref in self.aliases:
                    visit(ref)
            ordered.append(n)

        for n in self.aliases:
            visit(n)
        lines = []
        for n in ordered:
            typ = self.py_type(self.schemas[n], n)
            typ = re.sub(r"^Union\[(\w+), None\]$", r"Optional[\1]", typ)
            lines.append(f"{n} = {typ}")
        return lines

    # ------------------------------------------------------------------ types
    def literal(self, values: List[Any]) -> str:
        return "Literal[" + ", ".join(repr(v) for v in values) + "]"

    def py_type(self, schema: Any, hint: str) -> str:
        if "$ref" in schema:
            name = schema["$ref"].split("/")[-1]
            if name in self.literals or name in self.typeddicts or name in self.aliases:
                return name
            return "Any"
        if "enum" in schema:
            return self.literal(schema["enum"])
        if "oneOf" in schema or "anyOf" in schema:
            parts = [self.py_type(s, hint) for s in schema.get("oneOf", schema.get("anyOf", []))]
            return "Union[" + ", ".join(dict.fromkeys(parts)) + "]"
        typ = schema.get("type")
        if isinstance(typ, list):
            parts = [self.py_type({**schema, "type": t}, hint) for t in typ]
            return "Union[" + ", ".join(dict.fromkeys(parts)) + "]" if len(parts) > 1 else parts[0]
        if typ == "string":
            return "str"
        if typ == "integer":
            return "int"
        if typ == "number":
            return "float"
        if typ == "boolean":
            return "bool"
        if typ == "null":
            return "None"
        if typ == "array":
            return "List[" + self.py_type(schema.get("items", {}), hint) + "]"
        if typ == "object":
            extra = schema.get("additionalProperties")
            if isinstance(extra, dict):
                return "Dict[str, " + self.py_type(extra, hint) + "]"
            return "Dict[str, Any]"
        return "Any"

    def collect_literals(self) -> None:
        for name, schema in self.schemas.items():
            if "enum" in schema:
                self.literals[name] = schema["enum"]
            for prop, sub in (schema.get("properties") or {}).items():
                if "enum" in sub and len(sub["enum"]) > 0:
                    alias = name + pascal(prop)
                    self.literals.setdefault(alias, sub["enum"])

    def request_schemas(self) -> List[str]:
        # Strict request objects are the ones with additionalProperties: false.
        return [n for n, s in self.schemas.items() if s.get("additionalProperties") is False]

    def error_details(self) -> Dict[str, str]:
        return dict(self.schemas["Error"].get("x-error-details") or {})

    def detail_schemas(self) -> List[str]:
        """The x-error-details schemas plus the object schemas they reference (ModelRef)."""
        out: List[str] = []

        def visit(name: str) -> None:
            if name in out:
                return
            out.append(name)
            for ref in re.findall(r"#/components/schemas/(\w+)", repr(self.schemas[name])):
                if self.schemas[ref].get("type") == "object" and self.schemas[ref].get("properties"):
                    visit(ref)

        for name in self.error_details().values():
            visit(name)
        return out

    def typeddict(self, name: str) -> str:
        schema = self.schemas[name]
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        req = {k: v for k, v in props.items() if k in required}
        opt = {k: v for k, v in props.items() if k not in required}

        def fields(d: Dict[str, Any]) -> str:
            return ", ".join(f"{k!r}: {self.py_type(v, name + pascal(k))}" for k, v in d.items())

        doc = (schema.get("description") or "").strip().replace("\n", " ")
        lines = []
        lines.append(f"_{name}Required = TypedDict('_{name}Required', {{{fields(req)}}})")
        lines.append(f"_{name}Optional = TypedDict('_{name}Optional', {{{fields(opt)}}}, total=False)")
        lines.append("")
        lines.append("")
        lines.append(f"class {name}(_{name}Required, _{name}Optional):")
        fallback = f"The contract's ``{name}`` schema."
        lines.append(f'    """{doc}"""' if doc and len(doc) < 300 else f'    """{fallback}"""')
        return "\n".join(lines)

    # ------------------------------------------------------------------ tables
    def operations(self) -> List[str]:
        rows = []
        for path, item in self.spec["paths"].items():
            shared = [deref(self.spec, p) for p in item.get("parameters", [])]
            for method in METHODS:
                op = item.get(method)
                if not op:
                    continue
                params = shared + [deref(self.spec, p) for p in op.get("parameters", [])]
                security = op.get("security", self.spec.get("security", []))
                scopes = tuple(s for req in security for v in req.values() for s in v)
                idem = any(p["in"] == "header" and p["name"] == "Idempotency-Key" for p in params)
                codes = {str(k): tuple(v) for k, v in (op.get("x-error-codes") or {}).items()}
                rows.append(
                    f"    {op['operationId']!r}: Operation({method.upper()!r}, {path!r}, "
                    f"authenticated={bool(security)!r}, scopes={scopes!r}, idempotency_key={idem!r}, "
                    f"error_codes={codes!r}),"
                )
        return rows

    def error_codes(self) -> Dict[int, List[str]]:
        """Union of every operation's x-error-codes, per status."""
        table: Dict[int, List[str]] = {}
        for item in self.spec["paths"].values():
            for method in METHODS:
                op = item.get(method)
                for status, codes in ((op or {}).get("x-error-codes") or {}).items():
                    merged = table.setdefault(int(status), [])
                    merged.extend(c for c in codes if c not in merged)
        return dict(sorted(table.items()))

    def server_url(self) -> str:
        server = (self.spec.get("servers") or [{}])[0]
        url = server.get("url", "")
        for var, conf in (server.get("variables") or {}).items():
            url = url.replace("{" + var + "}", str(conf.get("default", "")))
        return url

    # ------------------------------------------------------------------ render
    def render(self, sha256: str) -> str:
        self.collect_literals()
        self.collect_aliases()
        reqs = self.request_schemas() + self.detail_schemas()
        self.typeddicts = list(reqs)
        info = self.spec["info"]
        license_info = info.get("license") or {}
        out: List[str] = []
        out.append('"""GENERATED by scripts/generate_types.py from sdks/openapi/bb-embed.v1.yaml.')
        out.append("")
        out.append("Do not edit by hand; run ``python scripts/generate_types.py``.")
        out.append('"""')
        out.append("")
        out.append("from __future__ import annotations")
        out.append("")
        out.append(
            "from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, TypedDict, Union"
        )
        out.append("")
        out.append(f"SPEC_VERSION = {info['version']!r}")
        out.append(f"SPEC_SHA256 = {sha256!r}")
        out.append(f"SPEC_LICENSE = {license_info.get('identifier') or license_info.get('name')!r}")
        out.append(f"DEFAULT_BASE_URL = {self.server_url()!r}")
        out.append("")
        out.append("# " + "-" * 66 + " enums")
        for name, values in sorted(self.literals.items()):
            out.append(f"{name} = {self.literal(values)}")
        out.append("")
        out.append("# " + "-" * 66 + " scalar and map aliases")
        out.extend(self.alias_lines())
        out.append("")
        out.append("# " + "-" * 66 + " request objects and error details")
        # Order so referenced TypedDicts are defined first.
        ordered: List[str] = []

        def visit(n: str) -> None:
            if n in ordered:
                return
            for v in (self.schemas[n].get("properties") or {}).values():
                for ref in re.findall(r"#/components/schemas/(\w+)", repr(v)):
                    if ref in reqs:
                        visit(ref)
            ordered.append(n)

        for n in reqs:
            visit(n)
        for n in ordered:
            out.append("")
            out.append("")
            out.append(self.typeddict(n))
        out.append("")
        out.append("")
        out.append("# " + "-" * 66 + " operations")
        out.append("class Operation(NamedTuple):")
        out.append("    method: str")
        out.append("    path: str")
        out.append("    authenticated: bool")
        out.append("    scopes: Tuple[str, ...]")
        out.append("    idempotency_key: bool")
        out.append("    error_codes: Dict[str, Tuple[str, ...]]")
        out.append("")
        out.append("")
        out.append("OPERATIONS: Dict[str, Operation] = {")
        out.extend(self.operations())
        out.append("}")
        out.append("")
        out.append("# Error codes per HTTP status: the union of every operation's x-error-codes.")
        out.append("ERROR_CODES: Dict[int, Tuple[str, ...]] = {")
        for status, codes in self.error_codes().items():
            out.append(f"    {status}: {tuple(codes)!r},")
        out.append("}")
        out.append("")
        out.append("# x-error-details: error code -> TypedDict of its `details`.")
        out.append("ERROR_DETAILS: Dict[str, Any] = {")
        for code, name in self.error_details().items():
            out.append(f"    {code!r}: {name},")
        out.append("}")
        text = "\n".join(out) + "\n"
        for kw in keyword.kwlist:  # sanity: keywords only ever appear as quoted keys
            assert f"\n    {kw}:" not in text
        return text


def main() -> int:
    raw = SPEC_PATH.read_bytes()
    text = Gen(yaml.safe_load(raw.decode("utf-8"))).render(hashlib.sha256(raw).hexdigest())
    if "--check" in sys.argv:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else None
        if current != text:
            print(
                f"stale: {OUT_PATH.relative_to(REPO_ROOT)}; run python scripts/generate_types.py",
                file=sys.stderr,
            )
            return 1
        print("generated types up to date")
        return 0
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
