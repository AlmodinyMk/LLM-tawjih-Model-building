"""Ranked and semantic search, export, pack, ingest, redact, stats and serve.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .core import (
    Die, FlowError, LEVELS, Node, SECTIONS, Store, __version__, die,
)
from .providers import (
    REDACT_TEMPLATE, Redactor, _post,
)
from .engine import (
    run_flow,
)
from .commands import (
    _flow_from_args,
)
from .library import (
    SOURCE_FORMATS, _tidy, cmd_deposit, from_html, source_to_text,
)
from .manage import (
    _passages, _paths_using, _place,
)


# ---------------------------------------------------------- ranked search
def _words(text):
    return [w.casefold() for w in re.findall(r"\w+", text) if len(w) > 1]


def _search_chunks(st, ids, sides):
    """Searchable pieces: the title, each reading passage, each paragraph
    of the other sides. [(id, side, text)]"""
    out = []
    for i in ids:
        n = st.nodes[i]
        out.append((i, "title", n.title))
        for s in sides:
            text = getattr(n, s)
            if not text.strip():
                continue
            pieces = _passages(text) if s == "reading" else \
                [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            for p in pieces:
                out.append((i, s, p.strip()))
    return out


def bm25(query, chunks, k1=1.5, b=0.75):
    """Okapi BM25 scores of chunks [(.., .., text)] for a query."""
    docs = [_words(c[2]) for c in chunks]
    if not docs:
        return []
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    q = _words(query)
    scores = []
    for d in docs:
        tf, s = {}, 0.0
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        for w in q:
            if w in tf:
                idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (
                    tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


# embeddings: a provider turns texts into vectors
def embed_ollama(texts, model):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return _post(host + "/api/embed", {"model": model, "input": texts},
                 {})["embeddings"]


def embed_openai(texts, model):
    """The same provider settings as the openai backend."""
    base = os.environ.get("TAWJIH_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TAWJIH_API_KEY", "")
    data = _post(base.rstrip("/") + "/embeddings",
                 {"model": model, "input": texts},
                 {"Authorization": "Bearer " + key} if key else {})["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


EMBEDDERS = {"ollama": embed_ollama, "openai": embed_openai}


def _cosine(u, v):
    num = sum(x * y for x, y in zip(u, v))
    den = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    return num / den if den else 0.0


def _index_file(st):
    return st.root / "index" / "embeddings.json"


def cmd_index(a):
    """Build or update the semantic index of the human and reading sides
    (and titles) with an embedding model. Only new or changed pieces are
    sent to the provider."""
    st = Store(a.project)
    model = a.model or os.environ.get("TAWJIH_EMBED_MODEL", "nomic-embed-text")
    f = _index_file(st)
    old = {}
    if f.is_file():
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        if data.get("backend") == a.backend and data.get("model") == model:
            old = {c["sha"]: c["vector"] for c in data["chunks"]}
    chunks = _search_chunks(st, st.subtree_ids(""), ["human", "reading"])
    rows, todo = [], []
    for i, side, text in chunks:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        rows.append({"id": i, "side": side, "sha": sha, "text": text[:4000]})
        if sha not in old:
            todo.append(rows[-1])
    print("%d piece(s), %d to embed with %s/%s" % (len(rows), len(todo),
                                                   a.backend, model),
          file=sys.stderr)
    for k in range(0, len(todo), 32):
        batch = todo[k:k + 32]
        try:
            vecs = EMBEDDERS[a.backend]([r["text"] for r in batch], model)
        except Exception as e:
            die("the embedding provider failed: %s: %s" % (type(e).__name__, e))
        for r, v in zip(batch, vecs):
            old[r["sha"]] = v
    for r in rows:
        r["vector"] = old[r["sha"]]
        r.pop("text")
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps({"backend": a.backend, "model": model,
                             "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "chunks": rows}), encoding="utf-8")
    print("index: %s (%d pieces)" % (f, len(rows)))


def search_ranked(st, query, ids, sides, semantic=False, top=10):
    chunks = _search_chunks(st, ids, sides)
    if semantic:
        f = _index_file(st)
        if not f.is_file():
            die("no semantic index yet; build it with: index --backend ollama")
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        vecs = {c["sha"]: c["vector"] for c in data["chunks"]}
        try:
            q = EMBEDDERS[data["backend"]]([query], data["model"])[0]
        except Exception as e:
            die("the embedding provider failed: %s: %s" % (type(e).__name__, e))
        scored, missing = [], 0
        for c in chunks:
            sha = hashlib.sha256(c[2].encode("utf-8")).hexdigest()[:16]
            if sha in vecs:
                scored.append((_cosine(q, vecs[sha]), c))
            elif c[1] in ("human", "reading", "title"):
                missing += 1
        if missing:
            print("note: %d piece(s) changed since the index was built; run "
                  "index again" % missing, file=sys.stderr)
    else:
        scored = list(zip(bm25(query, chunks), chunks))
        scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return scored[:top]


def cmd_search(a):
    """Search across trees or within a branch and side: literal by default,
    ranked by relevance with --rank (BM25), or by meaning with --semantic
    (an index built by the index command)."""
    st = Store(a.project)
    root = st.resolve(a.branch) if a.branch else ""
    ids = st.subtree_ids(root)
    if a.tag:
        ids = [i for i in ids if a.tag in st.nodes[i].tags]
    side_map = {"h": "human", "human": "human",
                "i": "instructions", "instructions": "instructions",
                "c": "code", "code": "code",
                "r": "reading", "reading": "reading"}
    if a.side and a.side.lower() not in side_map:
        die("unknown side '%s'" % a.side)
    target_sides = [side_map[a.side.lower()]] if a.side else SECTIONS
    if a.rank or a.semantic:
        hits = search_ranked(st, a.query, ids, target_sides, a.semantic,
                             a.top)
        for score, (i, side, text) in hits:
            print("%6.3f  %-8s [%s] %s  (%s)" % (score, st.number(i), i,
                                                 st.nodes[i].title, side))
            print("         " + " ".join(text.split())[:110])
        print("%d result(s)" % len(hits))
        return
    q = a.query.casefold()
    hits = 0
    for nid in ids:
        n = st.nodes[nid]
        matched = []
        if q in n.title.casefold() or q in nid.casefold():
            matched.append(("title", n.title))
        for sec in target_sides:
            text = getattr(n, sec)
            if q in text.casefold():
                line = next((l.strip() for l in text.splitlines()
                             if q in l.casefold()), "")
                matched.append((sec, line[:90]))
        if matched:
            hits += 1
            num = st.number(nid)
            print("%-8s [%s] %-8s %s" % (num, nid, LEVELS[n.level - 1],
                                         n.title))
            for sec, snippet in matched:
                print("         (%s) %s" % (sec, snippet))
    print("%d matching file(s)" % hits)


# ------------------------------------------------------------------ export
def _md_to_html(md):
    """The small Markdown that export writes: headings, paragraphs, '>'
    quotes, '- ' lists, fenced code. Mathematics stays as written."""
    import html as H

    def inline(t):
        t = H.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", t)
    out, para, quote, items, code = [], [], [], [], None

    def flush():
        if para:
            out.append('<p dir="auto">%s</p>' % inline(" ".join(para)))
            para.clear()
        if quote:
            paras, cur = [], []
            for q in quote + [""]:
                if q:
                    cur.append(inline(q))
                elif cur:
                    paras.append("<br>".join(cur))
                    cur = []
            out.append('<blockquote dir="auto">%s</blockquote>' % "".join(
                "<p>%s</p>" % p for p in paras))
            quote.clear()
        if items:
            out.append("<ul>%s</ul>" % "".join('<li dir="auto">%s</li>'
                                               % inline(x) for x in items))
            items.clear()
    for line in md.split("\n"):
        if code is not None:
            if line.startswith("```"):
                out.append("<pre><code>%s</code></pre>" % H.escape(
                    "\n".join(code)))
                code = None
            else:
                code.append(line)
            continue
        if line.startswith("```"):
            flush()
            code = []
            continue
        m = re.match(r"(#{1,6}) (.*)", line)
        if m:
            flush()
            out.append('<h%d dir="auto">%s</h%d>' % (
                len(m.group(1)), inline(m.group(2)), len(m.group(1))))
        elif line.startswith(">"):
            if para or items:
                flush()
            quote.append(line[1:].strip())
        elif line.startswith("- "):
            if para or quote:
                flush()
            items.append(line[2:])
        elif not line.strip():
            if quote:
                quote.append("")
            else:
                flush()
        else:
            if quote or items:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def cmd_export(a):
    """The trees (or one branch) as one readable book: the human side of
    each file and, unless left out, its reading passages with sources;
    --machine adds the machine side."""
    st = Store(a.project)
    root = st.resolve(a.branch) if a.branch else ""
    ids = st.subtree_ids(root)
    if not ids:
        die("nothing to export")
    top = min(st.nodes[i].level for i in ids)
    parts = []
    for i in ids:
        n = st.nodes[i]
        depth = min(6, n.level - top + 1)
        parts.append("%s %s %s" % ("#" * depth, st.number(i), n.title))
        if n.tags:
            parts.append("*%s*" % ", ".join(n.tags))
        if n.human.strip():
            parts.append(n.human.strip())
        if a.machine and (n.instructions.strip() or n.code.strip()):
            if n.instructions.strip():
                parts.append("**Instructions**\n\n" + n.instructions.strip())
            if n.code.strip():
                parts.append("```python\n%s\n```" % n.code.strip())
        if n.reading.strip() and not a.no_reading:
            for p in _passages(n.reading):
                lines = p.split("\n")
                if lines[0].startswith("## "):
                    lines[0] = "**%s**" % lines[0][3:]
                parts.append("\n".join("> " + l if l.strip() else ">"
                                       for l in lines))
    md = "\n\n".join(parts) + "\n"
    title = st.nodes[root].title if root else "Tawjih project"
    fmt = a.format or ("html" if (a.out or "").lower().endswith((".html",
                                                                 ".htm"))
                       else "md")
    if fmt == "html":
        import html as H
        mathjax = ""
        if not a.no_mathjax and re.search(r"\$[^$\n]+\$", md):
            # formulas are shown by MathJax, loaded from the web when the
            # page is opened; --no-mathjax keeps the page offline
            mathjax = ("<script>window.MathJax={tex:{inlineMath:[['$','$']]}};"
                       "</script><script defer src=\"https://cdn.jsdelivr.net/"
                       "npm/mathjax@3.2.2/es5/tex-chtml.js\"></script>")
        text = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, "
                "initial-scale=1\"><title>%s</title><style>body{max-width:"
                "46rem;margin:2rem auto;padding:0 1rem;font-family:serif;"
                "line-height:1.6}blockquote{border-inline-start:3px solid #bbb;"
                "margin:1rem 0;padding:0 1rem;color:#333}pre{background:#f4f4f4;"
                "padding:.8rem;overflow:auto}</style>%s</head><body>\n%s\n"
                "</body></html>\n" % (H.escape(title), mathjax,
                                       _md_to_html(md)))
    else:
        text = md
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("exported %d file(s) to %s (%s)" % (len(ids), a.out, fmt))
    else:
        sys.stdout.write(text)


# -------------------------------------------------------------- pack, unpack
def cmd_pack(a):
    """A branch (or the files of paths) with their sides, contracts and the
    paths among them, in one zip file for another project."""
    import zipfile
    st = Store(a.project)
    ids, names = [], list(a.path or [])
    for ref in a.branch or []:
        ids += [i for i in st.subtree_ids(st.resolve(ref)) if i not in ids]
    for name in names:
        ids += [i for i in _flow_from_args(st, name, []).nodes if i not in ids]
    if not ids:
        die("give --branch REF and/or --path NAME")
    for name, used in _paths_using(st, ids).items():
        try:
            if name not in names and set(st.load_flow(name).nodes) <= set(ids):
                names.append(name)
        except FlowError:
            pass
    roots = [i for i in ids if st.nodes[i].parent not in ids]
    manifest = {"tawjih": __version__, "created": time.strftime(
        "%Y-%m-%d %H:%M:%S"), "files": ids, "roots": roots,
        "paths": sorted(names)}
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                               indent=2))
        for i in ids:
            z.write(st.file_of(i), "files/%s.md" % i)
        for name in names:
            z.write(st.pdir / (name + ".path"), "paths/%s.path" % name)
    print("packed %d file(s) and %d path(s) into %s" % (len(ids), len(names),
                                                       out))


def cmd_unpack(a):
    """Add a pack to this project: its trees become new trees, or go under
    --parent. Ids that already exist are refused unless --new-ids."""
    import zipfile
    st = Store(a.project)
    try:
        z = zipfile.ZipFile(a.file)
    except (OSError, zipfile.BadZipFile) as e:
        die("not a pack: %s" % e)
    with z:
        bad = [n for n in z.namelist() if not re.fullmatch(
            r"manifest\.json|files/[0-9a-f]{6}\.md|paths/[\w\-]+\.path", n)]
        if bad:
            die("not a Tawjih pack (unexpected entries: %s)" % ", ".join(bad[:5]))
        try:
            man = json.loads(z.read("manifest.json"))
            files, pnames = list(man["files"]), list(man.get("paths", []))
        except (KeyError, ValueError, TypeError):
            die("the pack has no readable manifest.json")
        if not all(isinstance(i, str) and re.fullmatch(r"[0-9a-f]{6}", i)
                   for i in files) or not all(
                isinstance(p, str) and re.fullmatch(r"[\w\-]+", p)
                for p in pnames):
            die("the pack's manifest names files or paths that are not valid")
        nodes = {}
        for i in files:
            try:
                nodes[i] = Node.parse(z.read("files/%s.md" % i).decode("utf-8"))
            except (KeyError, ValueError) as e:
                die("pack file %s: %s" % (i, e))
        try:
            paths = {p: z.read("paths/%s.path" % p).decode("utf-8")
                     for p in pnames}
        except (KeyError, UnicodeDecodeError) as e:
            die("pack path: %s" % e)
    clash = [i for i in nodes if i in st.nodes]
    if clash and not a.new_ids:
        die("%d file id(s) already exist here (%s); add --new-ids to give the "
            "unpacked files new ones" % (len(clash), ", ".join(clash[:5])))
    mapping = {i: i for i in nodes}
    if a.new_ids:
        for i in nodes:
            nn = uuid.uuid4().hex[:6]
            while nn in st.nodes or nn in mapping.values():
                nn = uuid.uuid4().hex[:6]
            mapping[i] = nn
    parent = st.resolve(a.parent) if a.parent else ""
    roots = [i for i in nodes if nodes[i].parent not in nodes]
    for r in roots:
        delta = (st.nodes[parent].level + 1 - nodes[r].level) if parent else 0
        sub = [i for i in nodes if r in _chain(nodes, i)]
        if max(nodes[i].level for i in sub) + delta > len(LEVELS):
            die("%s would go deeper than level %d under that parent"
                % (nodes[r].title, len(LEVELS)))
        for i in sub:
            nodes[i].level += delta
    for i, n in nodes.items():
        n.id = mapping[i]
        n.parent = mapping[n.parent] if n.parent in nodes else (
            parent if i in roots else n.parent)
        n.links = [(mapping.get(t, t), ty) for t, ty in n.links]
        st.save(n)
    for r in roots:
        _place(st, mapping[r], parent)
    for name, text in paths.items():
        for old, new in mapping.items():
            text = re.sub(r"\b%s\b" % old, new, text)
        target, k = name, 1
        while (st.pdir / (target + ".path")).exists():
            k += 1
            target = "%s-%d" % (name, k)
        st.pdir.mkdir(exist_ok=True)
        (st.pdir / (target + ".path")).write_text(text, encoding="utf-8")
        print("path %s%s" % (target, "" if target == name else
                             " (renamed: %s exists)" % name))
    for r in roots:
        print("unpacked %s [%s] %s" % (st.number(mapping[r]), mapping[r],
                                       nodes[r].title))
    print("%d file(s) added" % len(nodes))


def _chain(nodes, i):
    seen = []
    while i in nodes and i not in seen:
        seen.append(i)
        i = nodes[i].parent
    return seen


# ------------------------------------------------------------------ ingest
def _docx_to_md(path):
    """Word .docx to Markdown: headings by style, lists, tables, text.
    Formulas (Office Math) are kept as their plain text between $ signs
    and counted, since they need checking."""
    import xml.etree.ElementTree as ET
    import zipfile
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    with zipfile.ZipFile(path) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        try:
            styles = ET.fromstring(z.read("word/styles.xml"))
        except KeyError:
            styles = None
    level_of = {}
    if styles is not None:
        for s in styles.iter(W + "style"):
            sid = s.get(W + "styleId", "")
            name = s.find(W + "name")
            name = name.get(W + "val", "") if name is not None else ""
            ol = s.find(".//" + W + "outlineLvl")
            m = re.search(r"(?:heading|عنوان)\s*(\d)", name, re.I) or \
                re.search(r"heading\s*(\d)", sid, re.I)
            if ol is not None:
                level_of[sid] = int(ol.get(W + "val", "0")) + 1
            elif m:
                level_of[sid] = int(m.group(1))
            elif name.lower() == "title" or sid.lower() == "title":
                level_of[sid] = 1
    formulas = [0]

    def para_text(p):
        parts = []
        for el in p.iter():
            if el.tag == M + "oMath":
                formulas[0] += 1
                parts.append("$%s$" % "".join(t.text or "" for t in
                                              el.iter(M + "t")))
            elif el.tag == W + "t" and not _inside(el, p, M + "oMath"):
                parts.append(el.text or "")
            elif el.tag == W + "tab":
                parts.append("\t")
            elif el.tag in (W + "br", W + "cr"):
                parts.append("\n")
        return "".join(parts)
    parents = {c: q for q in doc.iter() for c in q}

    def _inside(el, stop, tag):
        while el is not None and el is not stop:
            if el.tag == tag:
                return True
            el = parents.get(el)
        return False
    out = []
    body = doc.find(W + "body")
    for block in (body if body is not None else []):
        if block.tag == W + "p":
            text = para_text(block).strip()
            if not text:
                continue
            ps = block.find(W + "pPr/" + W + "pStyle")
            sid = ps.get(W + "val", "") if ps is not None else ""
            if sid in level_of:
                out.append("%s %s" % ("#" * min(6, level_of[sid]),
                                      " ".join(text.split())))
            elif block.find(W + "pPr/" + W + "numPr") is not None:
                out.append("- " + text)
            else:
                out.append(text)
        elif block.tag == W + "tbl":
            for row in block.iter(W + "tr"):
                cells = [" ".join(para_text(p).strip() for p in
                                  c.iter(W + "p")) for c in row.iter(W + "tc")]
                out.append("| %s |" % " | ".join(cells))
    md = ""
    for k, line in enumerate(out):
        both = k and out[k - 1].startswith(("- ", "| ")) and \
            line[:2] == out[k - 1][:2]
        md += ("\n" if both else "\n\n" if k else "") + line
    return _tidy(md), formulas[0]


def _epub_to_md(path):
    import xml.etree.ElementTree as ET
    import zipfile
    with zipfile.ZipFile(path) as z:
        cont = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in cont.iter()
                        if e.get("full-path"))
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        items = {e.get("id"): e.get("href") for e in opf.iter()
                 if e.tag.endswith("item")}
        spine = [e.get("idref") for e in opf.iter() if e.tag.endswith("itemref")]
        parts = []
        for ref in spine:
            href = items.get(ref)
            if href:
                try:
                    parts.append(from_html(z.read(base + href).decode(
                        "utf-8", "replace")))
                except KeyError:
                    pass
    return _tidy("\n\n".join(parts))


INGEST_TOOLS = {        # what each converter reads, best first per type
    ".pdf": ["docling", "marker", "pdftotext"],
    ".docx": ["builtin", "pandoc", "docling"],    # pandoc first if formulas
    ".epub": ["pandoc", "builtin"],
    ".odt": ["pandoc"], ".rtf": ["pandoc"], ".pptx": ["docling"],
}
INGEST_EXE = {"docling": "docling", "marker": "marker_single",
              "pandoc": "pandoc", "pdftotext": "pdftotext"}


def _docx_has_math(path):
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return b"oMath" in z.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile):
        return False


def _run_tool(cmd, what):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        die("%s could not start: %s" % (what, e))
    if r.returncode:
        die("%s failed: %s" % (what, (r.stderr or r.stdout).strip()[-800:]))


def _external(tool, src):
    """Markdown from an installed converter."""
    import tempfile
    exe = shutil.which(INGEST_EXE[tool])
    if not exe:
        die("%s is not installed" % tool)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if tool == "pandoc":
            out = tmp / "out.md"
            # plain Markdown: no direction spans or attributes around the
            # text, formulas as $...$ LaTeX
            _run_tool([exe, str(src), "-t",
                       "markdown-smart-bracketed_spans-native_spans-raw_html"
                       "-fenced_divs-native_divs-link_attributes"
                       "-raw_attribute", "--wrap=none", "-o", str(out)],
                      "pandoc")
        elif tool == "docling":
            _run_tool([exe, str(src), "--to", "md", "--output", str(tmp)],
                      "docling")
            out = next(tmp.rglob("*.md"), None)
        elif tool == "marker":
            _run_tool([exe, str(src), "--output_dir", str(tmp)], "marker")
            out = next(tmp.rglob("*.md"), None)
        else:
            out = tmp / "out.txt"
            _run_tool([exe, "-enc", "UTF-8", str(src), str(out)], "pdftotext")
        if out is None or not out.is_file():
            die("%s wrote no Markdown" % tool)
        text = out.read_text(encoding="utf-8", errors="replace")
    return _tidy(text.replace("\f", "\n\n"))


def cmd_ingest(a):
    """Turn an office document into Markdown for deposit. With --with auto
    the best converter installed is used: for PDF Docling, then Marker,
    then pdftotext; for Word and EPUB Pandoc, then the built-in reader."""
    src = Path(a.file)
    if not src.is_file():
        die("no such file: %s" % src)
    ext = src.suffix.lower()
    note = ""
    if ext in SOURCE_FORMATS and ext not in INGEST_TOOLS:
        md, used = source_to_text(src)[0], "builtin"
    else:
        if ext not in INGEST_TOOLS:
            die("ingest reads %s (and the deposit formats)" % ", ".join(
                sorted(INGEST_TOOLS)))
        tools = INGEST_TOOLS[ext]
        if ext == ".docx" and a.tool == "auto" and _docx_has_math(src):
            tools = ["pandoc", "builtin"]   # pandoc turns formulas into LaTeX
        if a.tool != "auto":
            if a.tool not in tools:
                die("%s does not read %s files; for them: %s" % (
                    a.tool, ext, ", ".join(tools)))
            tools = [a.tool]
        used = next((t for t in tools if t == "builtin"
                     or shutil.which(INGEST_EXE[t])), None)
        if used is None:
            die("no converter for %s is installed. Install one of: %s "
                "(Docling: pip install docling; Marker: pip install "
                "marker-pdf; pdftotext: poppler-utils; Pandoc: pandoc.org). "
                "Better still, look for the book's source (LaTeX, PreTeXt, "
                "HTML) and deposit that"
                % (ext, ", ".join(t for t in tools if t != "builtin")))
        if used == "builtin" and ext == ".docx":
            md, formulas = _docx_to_md(src)
            if formulas:
                note = ("%d formula(s) were flattened to plain text between $ "
                        "signs and are not usable as they are; install Pandoc "
                        "to get them as LaTeX" % formulas)
        elif used == "builtin":
            md = _epub_to_md(src)
        else:
            md = _external(used, src)
            if used == "pdftotext":
                note = ("pdftotext gives no headings and damages formulas "
                        "(exponents, fractions, arrows); add '#' headings and "
                        "check the mathematics before filtering, or install "
                        "Docling or Marker")
            elif ext == ".pdf":
                note = ("formulas read from a PDF by a model-based converter "
                        "can be wrong while looking right; check them")
    out = Path(a.out) if a.out else src.with_suffix(".md")
    if out.resolve() == src.resolve():
        die("the output would overwrite the input; give -o")
    out.write_text(md, encoding="utf-8")
    heads = sum(1 for l in md.split("\n") if re.match(r"#{1,6} \S", l))
    print("written %s with %s (%d characters, %d headings)" % (
        out, used, len(md), heads))
    if note:
        print("note: " + note)
    if a.deposit:
        a.file = str(out)
        a.part = None
        cmd_deposit(a)


# -------------------------------------------------------------- redact, stats
def cmd_redact(a):
    """Create redact.txt, or show what it hides in a text."""
    st = Store(a.project)
    f = st.root / "redact.txt"
    if a.init:
        if f.exists():
            die("redact.txt already exists")
        f.write_text(REDACT_TEMPLATE, encoding="utf-8")
        print("created %s: from now on every model call is redacted; edit "
              "it to add your own names and words" % f)
        return
    red = Redactor.load(st.root)
    if red is None:
        die("no redact.txt in this project (create it with: redact --init)")
    text = Path(a.file).read_text(encoding="utf-8-sig") if a.file else a.text
    if text is None:
        die("give a text, or --file")
    masked, mapping = red.mask(text)
    print(masked)
    for ph in sorted(mapping, key=lambda k: (k.rsplit("_", 1)[0],
                                             int(k.rsplit("_", 1)[1][:-1]))):
        print("  %s = %s" % (ph, mapping[ph]), file=sys.stderr)
    print("%d item(s) would be hidden" % len(mapping), file=sys.stderr)


def _prices(st):
    f = st.root / "prices.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        die("prices.json: %s" % e)


def cmd_stats(a):
    """What runs cost: per path the runs, how many finished, model calls,
    retries, fallbacks, answers from the cache, tokens and time; a cost when
    prices.json gives a price per million tokens for the model; and the
    latest evaluation."""
    st = Store(a.project)
    hdir = st.root / "history"
    since = time.time() - a.days * 86400 if a.days else 0
    rows, prices = {}, _prices(st)
    for f in sorted(hdir.glob("production-*.json")) if hdir.is_dir() else []:
        if f.stat().st_mtime < since:
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError:
            continue
        name = rec.get("name", "?")
        if a.path and name != a.path:
            continue
        r = rows.setdefault(name, {"runs": 0, "done": 0, "calls": 0,
                                   "retries": 0, "fallbacks": 0, "cached": 0,
                                   "in": 0, "out": 0, "seconds": 0.0,
                                   "cost": 0.0, "priced": True, "known": True})
        r["runs"] += 1
        r["done"] += rec.get("status") == "done"
        r["cached"] += rec.get("from_cache", 0)
        u = rec.get("usage")
        if not u:
            r["known"] = False         # a run made before 0.8
            continue
        for k in ("calls", "retries", "fallbacks", "in", "out", "seconds"):
            r[k] += u.get(k, 0)
        for p in rec.get("steps", []):
            pu = p.get("usage")
            if not pu:
                continue
            price = prices.get(p.get("model") or rec.get("model") or "")
            if price:
                r["cost"] += (pu.get("in", 0) * price.get("input", 0)
                              + pu.get("out", 0) * price.get("output", 0)) / 1e6
            elif pu.get("in") or pu.get("out"):
                r["priced"] = False
    evals = {}
    for f in sorted(st.rdir.glob("eval-*.json"), key=_by_time) \
            if st.rdir.is_dir() else []:
        try:
            e = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError:
            continue
        if e.get("mode") == "direct":
            continue
        evals[e.get("path")] = (e.get("passed", 0), e.get("total", 0),
                                f.stem[-15:])
    if not rows:
        print("no runs recorded%s" % (" for path " + a.path if a.path else ""))
    for name in sorted(rows):
        r = rows[name]
        cost = ("%.4f" % r["cost"]) if r["priced"] and prices else "-"
        print("%s: %d run(s), %d finished; %d model call(s), %d retried, %d "
              "fallback(s), %d answer(s) from the cache" % (
                  name, r["runs"], r["done"], r["calls"], r["retries"],
                  r["fallbacks"], r["cached"]))
        print("  tokens in %d, out %d; model time %.1f s; cost %s%s" % (
            r["in"], r["out"], r["seconds"], cost,
            "" if r["known"] else "  (runs before 0.8 kept no usage)"))
        if name in evals:
            p_, t_, when = evals[name]
            print("  latest evaluation: %d of %d passed (%s)" % (p_, t_, when))
    if not prices:
        print("costs: write prices.json with the price per million tokens, "
              'e.g. {"MODEL": {"input": 3.0, "output": 15.0}}; local models '
              "cost nothing to call")
    print("tokens are those the provider reports; manual and dry report none")


# -------------------------------------------------------------------- serve
def cmd_serve(a):
    """Offer saved paths to other programs over HTTP on this computer.
    POST /paths/NAME with {"input": "..."} and the token runs the path and
    returns its result; GET /paths lists them. One run at a time."""
    import http.server
    import secrets
    import threading
    st = Store(a.project)
    token = a.token or os.environ.get("TAWJIH_TOKEN") or secrets.token_urlsafe(18)
    allowed = set(a.paths.split(",")) if a.paths else None
    lock = threading.Lock()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: serving on %s makes the paths reachable from other "
              "computers" % a.host, file=sys.stderr)

    def path_names():
        names = sorted(f.stem for f in st.pdir.glob("*.path"))
        return [n for n in names if allowed is None or n in allowed]

    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token)

        def log_message(self, fmt, *args):
            print("serve: " + fmt % args, file=sys.stderr)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"tawjih": __version__})
            if not self.authorized():
                return self.send(401, {"error": "missing or wrong token"})
            if self.path == "/paths":
                fresh = Store(a.project)
                out = []
                for name in path_names():
                    try:
                        flow = fresh.load_flow(name)
                    except FlowError as e:
                        out.append({"name": name, "error": str(e)})
                        continue
                    out.append({"name": name, "steps": len(flow.nodes),
                                "code": any(fresh.nodes[i].code.strip()
                                            for i in flow.nodes)})
                return self.send(200, out)
            self.send(404, {"error": "unknown address"})

        def do_POST(self):
            if not self.authorized():
                return self.send(401, {"error": "missing or wrong token"})
            m = re.fullmatch(r"/paths/([\w\-]+)", self.path)
            if not m or m.group(1) not in path_names():
                return self.send(404, {"error": "no such path"})
            size = int(self.headers.get("Content-Length", "0") or 0)
            if size > 1_000_000:
                return self.send(413, {"error": "the request is too large"})
            try:
                req = json.loads(self.rfile.read(size) or b"{}")
                text = req["input"]
                if not isinstance(text, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self.send(400, {"error": 'send JSON: {"input": "..."}'})
            name = m.group(1)
            with lock:
                fresh = Store(a.project)
                try:
                    flow = fresh.load_flow(name)
                except FlowError as e:
                    return self.send(500, {"error": str(e)})
                if any(fresh.nodes[i].code.strip() for i in flow.nodes) \
                        and not a.allow_code:
                    return self.send(403, {"error": "this path has code; "
                                           "start serve with --allow-code"})
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err):
                        result = run_flow(fresh, name, text, a.backend,
                                          a.allow_code, flow,
                                          resume=a.resume,
                                          code_timeout=a.code_timeout,
                                          retries=a.retries,
                                          approve=a.approve or "deny",
                                          sandbox=getattr(a, "sandbox", False))
                except Die as e:
                    return self.send(500, {"error": str(e.code),
                                           "messages": err.getvalue()})
                except Exception as e:          # a fault in the program
                    return self.send(500, {"error": "%s: %s" % (
                        type(e).__name__, e), "messages": err.getvalue()})
                return self.send(200, {"path": name, "result": result,
                                       "messages": err.getvalue()})

    server = http.server.ThreadingHTTPServer((a.host, a.port), Handler)
    host, port = server.server_address[:2]
    print("serving %d path(s) on http://%s:%d  (backend %s%s)" % (
        len(path_names()), host, port, a.backend,
        ", code allowed" if a.allow_code else ""))
    print("token: %s" % token)
    print('try: curl -H "Authorization: Bearer %s" http://%s:%d/paths'
          % (token, host, port))
    if getattr(a, "_ready", None):
        a._ready(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _by_time(f):
    """Sort key for stamped files: name-20261120-093058[-k].json."""
    m = re.search(r"(\d{8}-\d{6})(?:-(\d+))?\.\w+$", f.name)
    return (m.group(1), int(m.group(2) or 1)) if m else ("", 0)


__all__ = [
    'EMBEDDERS', 'INGEST_EXE', 'INGEST_TOOLS', '_by_time', '_chain',
    '_cosine', '_docx_has_math', '_docx_to_md', '_epub_to_md',
    '_external', '_index_file', '_md_to_html', '_prices', '_run_tool',
    '_search_chunks', '_words', 'bm25', 'cmd_export', 'cmd_index',
    'cmd_ingest', 'cmd_pack', 'cmd_redact', 'cmd_search', 'cmd_serve',
    'cmd_stats', 'cmd_unpack', 'embed_ollama', 'embed_openai',
    'search_ranked',
]
