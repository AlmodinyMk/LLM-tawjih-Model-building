"""The library: deposit, filter, accept and the source-format readers (Markdown, LaTeX, HTML, PreTeXt).

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from .core import (
    LEVELS, MARK, SECTIONS, Store, die,
)
from .providers import (
    call_model, use_model,
)
from .commands import (
    _backup,
)


# ------------------------------------------------ library: deposit + filter
def _library(st):
    ldir = st.root / "library"
    ldir.mkdir(exist_ok=True)
    idx = ldir / "index.json"
    books = json.loads(idx.read_text(encoding="utf-8")) if idx.is_file() else {}
    return ldir, idx, books


def _book(st, ref):
    ldir, _, books = _library(st)
    hits = [b for b in books if b.startswith(ref)]
    if len(hits) != 1:
        die("no single deposited book matches '%s' (see: library)" % ref)
    return ldir, hits[0], books[hits[0]]


# ------------------------------------------------------------ source formats
# deposit turns a source file into plain text with '#' headings (one '#' per
# level), which is what filter splits on. Mathematics is kept as LaTeX
# ($...$ inline, $$...$$ displayed). Standard library only.

def _tidy(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _heading(level, title):
    title = " ".join(title.split())
    return "\n\n%s %s\n\n" % ("#" * max(1, min(level, 6)), title)


def from_markdown(text, path=None):
    """Markdown already uses '#' headings. Setext headings become '#', and
    '#' lines inside code fences are protected."""
    out, fence = [], None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*(```+|~~~+)", line)
        if m:
            mark = m.group(1)[0] * 3
            fence = None if fence and mark == fence else (fence or mark)
            out.append(line)
        elif fence:
            out.append(" " + line if line.startswith("#") else line)
        elif (i + 1 < len(lines) and line.strip()
              and re.fullmatch(r"\s*(=+|-+)\s*", lines[i + 1])
              and not line.lstrip().startswith(("-", "*", "#", ">"))):
            out.append("%s %s" % ("#" if "=" in lines[i + 1] else "##",
                                  line.strip()))
            i += 1
        else:
            m = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
            out.append("%s %s" % (m.group(1), m.group(2)) if m else line)
        i += 1
    return _tidy("\n".join(out))


TEX_LEVELS = ["part", "chapter", "section", "subsection", "subsubsection",
              "paragraph", "subparagraph"]
TEX_BLOCKS = {"theorem": "Theorem", "lemma": "Lemma", "proposition": "Proposition",
              "corollary": "Corollary", "definition": "Definition",
              "example": "Example", "remark": "Remark", "proof": "Proof",
              "exercise": "Exercise", "conjecture": "Conjecture",
              "claim": "Claim", "fact": "Fact", "note": "Note",
              "problem": "Problem", "solution": "Solution", "axiom": "Axiom",
              "notation": "Notation", "question": "Question"}
TEX_VERBATIM = r"verbatim\*?|Verbatim\*?|BVerbatim|LVerbatim|lstlisting|minted|alltt"
TEX_MATH = (r"equation|align|gather|multline|eqnarray|displaymath|flalign|"
            r"alignat|IEEEeqnarray|dmath|math")
TEX_DROP = r"tikzpicture|tikzcd|pspicture|picture|filecontents\*?|comment"
TEX_LAYOUT = (r"itshape|bfseries|slshape|scshape|upshape|mdseries|rmfamily|"
              r"sffamily|ttfamily|normalfont|em|it|bf|sl|sc|tt|rm|sf|tiny|"
              r"scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|"
              r"Huge|centering|raggedright|raggedleft|noindent|indent|par|"
              r"newline|linebreak|pagebreak|nolinebreak|nopagebreak|clearpage|"
              r"cleardoublepage|newpage|smallskip|medskip|bigskip|vfill|hfill|"
              r"null|maketitle|tableofcontents|listoffigures|listoftables|"
              r"appendix|frontmatter|mainmatter|backmatter|protect|relax|"
              r"centerline|leavevmode|hline|toprule|midrule|bottomrule|"
              r"addlinespace|big|Big|bigg|Bigg")
TEX_NOTTEXT = (r"index|glossary|vspace|hspace|addvspace|setlength|addtolength|"
               r"addtocounter|setcounter|stepcounter|refstepcounter|"
               r"addcontentsline|markboth|markright|bibliographystyle|"
               r"bibliography|usepackage|RequirePackage|documentclass|"
               r"pagestyle|thispagestyle|pagenumbering|color|graphicspath|"
               r"input|include|includeonly|hypersetup|geometry|setcopyright|"
               r"selectlanguage|indexsee|cline|cmidrule")
TEX_STYLE = (r"emph|textbf|textit|textsl|textsc|texttt|textsf|textrm|textup|"
             r"textmd|textnormal|underline|uline|mbox|hbox|fbox|makebox|text|"
             r"textsuperscript|textsubscript|enquote|textquote|"
             r"MakeUppercase|MakeLowercase|uppercase|lowercase|"
             r"foreignlanguage\s*\{[^}]*\}|textcolor\s*\{[^}]*\}|"
             r"colorbox\s*\{[^}]*\}")
TEX_ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308",
               "~": "\u0303", "=": "\u0304", ".": "\u0307", "u": "\u0306",
               "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
               "d": "\u0323", "b": "\u0331", "r": "\u030a", "t": "\u0361"}
TEX_WORDS = {"ldots": "\u2026", "dots": "\u2026", "textellipsis": "\u2026",
             "LaTeXe": "LaTeX2e", "LaTeX": "LaTeX", "TeX": "TeX",
             "XeLaTeX": "XeLaTeX", "LuaLaTeX": "LuaLaTeX", "today": "",
             "slash": "/", "textbackslash": "\\", "ss": "\u00df",
             "o": "\u00f8", "O": "\u00d8", "ae": "\u00e6", "AE": "\u00c6",
             "oe": "\u0153", "OE": "\u0152", "aa": "\u00e5", "AA": "\u00c5",
             "l": "\u0142", "L": "\u0141", "i": "\u0131", "j": "\u0237",
             "dag": "\u2020", "ddag": "\u2021", "S": "\u00a7", "P": "\u00b6",
             "copyright": "\u00a9", "pounds": "\u00a3", "texteuro": "\u20ac",
             "euro": "\u20ac", "textregistered": "\u00ae",
             "texttrademark": "\u2122", "textdegree": "\u00b0",
             "textendash": "\u2013", "textemdash": "\u2014",
             "textquoteleft": "\u2018", "textquoteright": "\u2019",
             "textquotedblleft": "\u201c", "textquotedblright": "\u201d",
             "quad": " ", "qquad": " ", "enspace": " ", "thinspace": " ",
             "textvisiblespace": "\u2423", "textbar": "|", "textless": "<",
             "textgreater": ">", "textasciitilde": "~", "textasciicircum": "^",
             "textunderscore": "_", "checkmark": "\u2713"}


def _tex_arg(s, i):
    """The {...} group starting at s[i] (after optional spaces): (text, end),
    or (None, i) when there is none."""
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i >= len(s) or s[i] != "{":
        return None, i
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _tex_strip_comments(t):
    """'%' to the end of the line, and with it the line break and the
    leading spaces of the next line; '\\%' is a percent sign."""
    return re.sub(r"(?<!\\)%[^\n]*\n?[ \t]*", "", t)


def _tex_includes(text, path, seen):
    """Read in the files named by \\input and \\include, recursively,
    except on lines where they are commented out."""
    path = Path(path).resolve()
    seen.add(path)

    def include(m):
        f = (path.parent / m.group(2).strip()).resolve()
        if not f.suffix:
            f = f.with_suffix(".tex")
        if f in seen or not f.is_file():
            return ""
        return _tex_includes(f.read_text(encoding="utf-8", errors="replace"),
                             f, seen)
    lines = []
    for line in text.split("\n"):
        c = re.search(r"(?<!\\)%", line)
        head, tail = (line[:c.start()], line[c.start():]) if c else (line, "")
        lines.append(re.sub(r"\\(input|include)\s*\{([^}]*)\}", include, head)
                     + tail)
    return "\n".join(lines)


def _tex_macros(s):
    """Read the definitions of the document's own commands and theorem
    environments, remove them, and return (text, macros, theorems)."""
    macros, theorems, out, i = {}, {}, [], 0
    defn = re.compile(r"\\(newcommand|renewcommand|providecommand|"
                      r"DeclareRobustCommand|DeclareMathOperator|newtheorem|"
                      r"def|NewDocumentCommand|RenewDocumentCommand|"
                      r"ProvideDocumentCommand)\*?\s*")

    def name_at(k):
        nm, k2 = _tex_arg(s, k)
        if nm is None:
            m = re.match(r"\s*(\\[A-Za-z@]+)", s[k:])
            nm, k2 = m.group(1), k + m.end()
        return nm.strip().lstrip("\\"), k2

    def opt(k):
        m = re.match(r"\s*\[([^\]]*)\]", s[k:])
        return (m.group(1), k + m.end()) if m else (None, k)

    while True:
        m = defn.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        kind, k = m.group(1), m.end()
        try:
            if kind == "def":
                nm = re.match(r"\\([A-Za-z@]+)([^{]*)", s[k:])
                body, k = _tex_arg(s, k + nm.end())
                if body is None or "#" in nm.group(2):
                    raise ValueError
                macros[nm.group(1)] = (0, None, body)
            elif kind == "newtheorem":
                env, k = _tex_arg(s, k)
                _, k = opt(k)
                label, k = _tex_arg(s, k)
                _, k = opt(k)
                theorems[env.strip()] = label.strip()
            elif kind.endswith("DocumentCommand"):
                nm, k = name_at(k)
                spec, k = _tex_arg(s, k)
                body, k = _tex_arg(s, k)
                macros[nm] = (len(re.findall(r"[mo]", spec or "")), None, body)
            elif kind == "DeclareMathOperator":
                nm, k = name_at(k)
                body, k = _tex_arg(s, k)
                macros[nm] = (0, None, "\\operatorname{%s}" % body)
            else:
                nm, k = name_at(k)
                n, k = opt(k)
                default, k = opt(k)
                body, k = _tex_arg(s, k)
                if body is None:
                    raise ValueError
                macros[nm] = (int(n or 0), default, body)
            i = k
        except (ValueError, AttributeError, TypeError):
            out.append(m.group(0))              # not understood: left as is
            i = m.end()
    return "".join(out), macros, theorems


def _tex_expand(t, macros, depth=0):
    if depth > 5 or not macros:
        return t
    pat = re.compile(r"\\(%s)(?![A-Za-z@])" % "|".join(
        re.escape(n) for n in sorted(macros, key=len, reverse=True)))
    res, i = [], 0
    for m in pat.finditer(t):
        if m.start() < i:
            continue
        n, default, body = macros[m.group(1)]
        k, args = m.end(), []
        if default is not None:
            o = re.match(r"\s*\[([^\]]*)\]", t[k:])
            args.append(o.group(1) if o else default)
            k += o.end() if o else 0
        while len(args) < n:
            g, k2 = _tex_arg(t, k)
            if g is None:
                one = re.match(r"\s*(\\[A-Za-z]+|\S)", t[k:])
                if not one:
                    break
                g, k2 = one.group(1), k + one.end()
            args.append(g)
            k = k2
        for a_i, a in enumerate(args, 1):
            body = body.replace("#%d" % a_i, a)
        res += [t[i:m.start()], body]
        i = k
    res.append(t[i:])
    t2 = "".join(res)
    return t2 if t2 == t else _tex_expand(t2, macros, depth + 1)


def from_latex(text, path=None):
    """LaTeX, in stages: code is protected first, then the document body is
    taken (an example document inside the text does not cut it short),
    \\input and \\include are read in, the document's own commands are
    expanded, mathematics is protected, and only then are sections,
    environments, references, text commands and special characters turned
    into text; code and mathematics are put back last."""
    kept = []
    if path:
        text = _tex_includes(text, path, set())

    def keep(kind, body):
        kept.append((kind, body))
        return "\ue000%d\ue001" % (len(kept) - 1)

    # 1. code, printed exactly as typed
    def verb_env(m):
        name, body = m.group(1), m.group(2)
        if name == "minted":
            body = re.sub(r"^\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", body, count=1)
        elif name.startswith(("lstlisting", "Verbatim", "BVerbatim",
                              "LVerbatim")):
            body = re.sub(r"^\s*\[[^\]]*\]", "", body, count=1)
        return "\n\n" + keep("code", body.strip("\n")) + "\n\n"
    s = re.sub(r"\\begin\{(%s)\}(.*?)\\end\{\1\}" % TEX_VERBATIM, verb_env,
               text, flags=re.S)
    s = re.sub(r"\\verb\*?([^A-Za-z*\s])(.*?)\1",
               lambda m: keep("inline", m.group(2)), s)
    s = re.sub(r"\\(lstinline|mintinline\s*\{[^}]*\})\s*(\[[^\]]*\])?"
               r"(?:\{([^}]*)\}|([^A-Za-z\s{])(.*?)\4)",
               lambda m: keep("inline", m.group(3) if m.group(3) is not None
                              else m.group(5)), s)

    # 2. comments; the document's own commands, defined in the preamble
    # too; then the document body
    s = _tex_strip_comments(s)
    s, macros, theorems = _tex_macros(s)
    dc = re.search(r"\\documentclass", s)
    first = re.search(r"\\(begin\{(?!document)|part\b|chapter\b|section\b)", s)
    if dc and not (first and first.start() < dc.start()):
        b, e = s.find("\\begin{document}", dc.end()), s.rfind("\\end{document}")
        if b >= 0:
            s = s[b + len("\\begin{document}"):e if e > b else None]
    else:
        s = re.sub(r"\\(begin|end)\{document\}", "", s)

    # 3. the document's own commands expanded
    s = _tex_expand(s, macros)

    # 4. mathematics, kept as LaTeX
    def math(kind, body):
        body = re.sub(r"\\label\s*\{[^}]*\}|\\nonumber\b|\\notag\b", "", body)
        return keep(kind, body.strip())
    s = re.sub(r"\\begin\{((?:%s)\*?)\}(.*?)\\end\{\1\}" % TEX_MATH,
               lambda m: "\n" + math("display", "\\begin{%s}%s\\end{%s}" % (
                   m.group(1), m.group(2), m.group(1))) + "\n", s, flags=re.S)
    for pat, kind in ((r"\$\$(.+?)\$\$", "display"), (r"\\\[(.+?)\\\]", "display"),
                      (r"\\\((.+?)\\\)", "math"),
                      (r"(?<!\\)\$(.+?)(?<!\\)\$", "math")):
        s = re.sub(pat, lambda m, k=kind: (("\n%s\n" if k == "display" else "%s")
                                           % math(k, m.group(1))), s, flags=re.S)

    # 5. sections
    used = [k for k in TEX_LEVELS if re.search(r"\\%s\*?\s*[\[{]" % k, s)]
    out, i = [], 0
    for m in re.finditer(r"\\(%s)\*?\s*(\[[^\]]*\])?\s*(?=\{)"
                         % "|".join(TEX_LEVELS), s):
        if m.start() < i:
            continue
        title, end = _tex_arg(s, m.end())
        if title is None:
            continue
        out.append(s[i:m.start()])
        out.append("\n\n\ue002%d %s\ue003\n\n" % (
            min(6, used.index(m.group(1)) + 1), " ".join(title.split())))
        i = end
    s = "".join(out) + s[i:]

    # 6. environments
    blocks = dict(TEX_BLOCKS, **theorems)
    s = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % TEX_DROP, "", s, flags=re.S)
    s = re.sub(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{([^}]*)\}",
               r"[image: \2]", s)
    s = re.sub(r"\\caption\s*(\[[^\]]*\])?\s*\{", "\n\nCaption: {", s)
    s = re.sub(r"\\begin\{(tabular\*?|tabularx|longtable|array|minipage|"
               r"wrapfigure)\}\s*(\[[^\]]*\])?\s*(\{(?:[^{}]|\{[^{}]*\})*\})?"
               r"(\s*\{(?:[^{}]|\{[^{}]*\})*\})?", "\n\n", s)

    def begin(m):
        base, title = m.group(1).rstrip("*"), m.group(3)
        if base in blocks:
            return "\n\n%s%s.\n" % (blocks[base], " (%s)" % title if title else "")
        if base in ("itemize", "enumerate", "description", "list"):
            return "\n"
        if title and base not in ("figure", "table", "center", "quote",
                                  "quotation", "verse"):
            return "\n\n%s (%s).\n" % (base.capitalize(), title)
        return "\n\n"
    s = re.sub(r"\\begin\{([A-Za-z*]+)\}(\[([^\]]*)\])?", begin, s)
    s = re.sub(r"\\end\{[A-Za-z*]+\}", "\n\n", s)
    s = re.sub(r"\\item\s*\[([^\]]*)\]", r"\n- \1: ", s)
    s = re.sub(r"\\item\b\s*", "\n- ", s)
    s = re.sub(r"(?<!\\)&", " | ", s)

    # 7. references and citations
    s = re.sub(r"\\label\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\(eqref|ref|pageref|autoref|cref|Cref|nameref|vref|vpageref|"
               r"labelcref|crefrange)\*?\s*\{([^}]*)\}(\s*\{([^}]*)\})?",
               lambda m: "[ref: %s]" % (m.group(2) + (
                   "\u2013" + m.group(4) if m.group(4) else "")), s)
    s = re.sub(r"\\(cite|citep|citet|citealp|citealt|parencite|textcite|"
               r"autocite|footcite|citeauthor|citeyear|nocite)\*?\s*"
               r"(\[[^\]]*\]\s*){0,2}\{([^}]*)\}", r"[cite: \3]", s)
    s = re.sub(r"\\href\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\2 (\1)", s)
    s = re.sub(r"\\(url|path)\s*\{([^}]*)\}", r"\2", s)

    # 8. text commands
    for _ in range(3):
        s = re.sub(r"\\footnote\s*(\[[^\]]*\])?\s*\{([^{}]*(\{[^{}]*\}[^{}]*)*)\}",
                   r" [footnote: \2]", s)
    for _ in range(4):
        s = re.sub(r"\\(%s)\*?\s*\{([^{}]*)\}" % TEX_STYLE, r"\2", s)
    s = re.sub(r"\\(%s)\*?\s*(\[[^\]]*\])?((\s*\{[^{}]*\})*)" % TEX_NOTTEXT, " ", s)
    s = re.sub(r"\\(%s)(?![A-Za-z])\s?" % TEX_LAYOUT, " ", s)

    # 9. special characters and accents
    def accent(m):
        ch = m.group(2) or m.group(3)
        ch = {"\\i": "i", "\\j": "j"}.get(ch, ch)
        return unicodedata.normalize("NFC", ch + TEX_ACCENTS[m.group(1)])
    s = re.sub(r"\\(['`^\"~=.]|[uvHckdbrt](?![A-Za-z]))\s*(?:\{(\\?[A-Za-z])\}|"
               r"(\\[ij](?![A-Za-z])|[A-Za-z]))", accent, s)
    s = re.sub(r"\\(%s)(?![A-Za-z])(\{\})?" % "|".join(
        sorted(TEX_WORDS, key=len, reverse=True)), lambda m: TEX_WORDS[m.group(1)], s)
    s = re.sub(r"\\([%&_#${}])", r"\1", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\\\\*?(\[[^\]]*\])?", "\n", s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"(?<!\\)~", " ", s)
    # a command still left is the document's own: its arguments are kept
    # (several are separated by spaces), or its name when it has none
    for _ in range(3):
        s = re.sub(r"\\([A-Za-z@]+)\*?\s*(\[[^\]]*\])?((\s*\{[^{}]*\})+)",
                   lambda m: " ".join(re.findall(r"\{([^{}]*)\}", m.group(3))), s)
    s = re.sub(r"\\([A-Za-z@]+)\*?", r"\1", s)
    s = re.sub(r"(?<=[A-Za-z])\{(?=[A-Za-z])", " ", s)  # a!!a{word}: no gluing
    s = re.sub(r"(?<!\\)[{}]", "", s)

    # 10. code, mathematics and headings back in place
    def put(m):
        kind, body = kept[int(m.group(1))]
        return {"code": "\n```\n%s\n```\n", "inline": "`%s`",
                "display": "\n$$%s$$\n", "math": "$%s$"}[kind] % body
    for _ in range(3):
        s = re.sub("\ue000(\\d+)\ue001", put, s)
    s = re.sub("\ue002(\\d) ([^\ue003]*)\ue003",
               lambda m: "\n\n%s %s\n\n" % ("#" * int(m.group(1)),
                                            m.group(2).strip()), s)
    return _tidy(_guard_hashes_body(s))



def _guard_hashes_body(text):
    """Like _guard_hashes, but keeps the headings this module wrote, which
    always stand alone between blank lines."""
    lines = text.split("\n")
    out = []
    for k, l in enumerate(lines):
        alone = ((k == 0 or not lines[k - 1].strip())
                 and (k + 1 == len(lines) or not lines[k + 1].strip()))
        out.append(l if not l.startswith("#") or (alone and re.match(
            r"#{1,6} \S", l)) else " " + l)
    return "\n".join(out)


class _HTMLText:
    """Collect the text of an HTML page with '#' headings."""
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg",
            "button", "form", "template"}
    BLOCK = {"p", "div", "section", "article", "blockquote", "pre", "table",
             "tr", "ul", "ol", "dl", "dd", "dt", "figure", "figcaption",
             "main", "aside", "br", "hr"}

    def __init__(self):
        import html.parser
        outer = self

        class P(html.parser.HTMLParser):
            def handle_starttag(self, tag, attrs):
                outer.start(tag, dict(attrs))

            def handle_endtag(self, tag):
                outer.end(tag)

            def handle_data(self, data):
                outer.data(data)

        self.parser = P(convert_charrefs=True)
        self.out, self.skip, self.head, self.math = [], 0, None, None

    def start(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":
            self.math = {"tex": attrs.get("alttext"), "text": [],
                         "display": attrs.get("display") == "block",
                         "ann": False}
        elif self.math is not None and tag == "annotation" and \
                "tex" in (attrs.get("encoding") or ""):
            self.math["ann"], self.math["tex"] = True, ""
        elif re.fullmatch(r"h[1-6]", tag):
            self.head = (int(tag[1]), [])
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag != "br" else "\n")

    def end(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math" and self.math is not None:
            m, self.math = self.math, None
            tex = (m["tex"] or "".join(m["text"])).strip()
            self.out.append(("\n$$%s$$\n" if m["display"] else "$%s$") % tex)
        elif tag == "annotation" and self.math is not None:
            self.math["ann"] = False
        elif self.head and tag == "h%d" % self.head[0]:
            level, words = self.head
            self.head = None
            self.out.append(_heading(level, "".join(words)))
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def data(self, data):
        if self.skip:
            return
        if self.math is not None:
            if self.math["ann"]:
                self.math["tex"] += data
            elif not self.math["tex"]:
                self.math["text"].append(data)
            return
        (self.head[1] if self.head else self.out).append(data)


def from_html(text, path=None):
    """HTML: <h1>-<h6> become headings; scripts, styles and navigation are
    left out; MathML is replaced by its TeX annotation or alttext; MathJax
    source (\\( \\), $$ $$) is text already."""
    h = _HTMLText()
    h.parser.feed(text)
    h.parser.close()
    raw = "".join(h.out)
    lines = []
    for l in raw.split("\n"):
        lines.append(l if re.match(r"#{1,6} \S", l) else " ".join(l.split()))
    return _tidy(_guard_hashes_body("\n".join(lines)))


PTX_DIVISIONS = {"part": 1, "chapter": 1, "appendix": 1, "section": 2,
                 "exercises": 2, "references": 2, "reading-questions": 2,
                 "glossary": 2, "solutions": 2, "worksheet": 2,
                 "subsection": 3, "subsubsection": 4, "paragraphs": 5}
PTX_BLOCKS = {"theorem", "lemma", "proposition", "corollary", "definition",
              "example", "remark", "proof", "exercise", "algorithm", "claim",
              "fact", "axiom", "conjecture", "note", "observation", "warning",
              "insight", "question", "problem", "activity", "exploration",
              "hint", "answer", "solution", "assumption", "principle",
              "identity", "heuristic", "hypothesis", "objectives",
              "outcomes", "project", "investigation", "computation",
              "technology", "data", "convention", "notation-block"}
PTX_SKIP = {"sage", "idx", "notation", "latex-image", "image", "index",
            "docinfo", "macros", "latex-image-preamble", "video",
            "interactive", "program", "console", "sageplot", "asymptote",
            "frontmatter", "backmatter", "bookinfo"}
PTX_EMPTY = {"mdash": "\u2014", "ndash": "\u2013", "ellipsis": "\u2026",
             "nbsp": " ", "tex": "TeX", "latex": "LaTeX", "pretext": "PreTeXt",
             "xetex": "XeTeX", "ie": "i.e.", "eg": "e.g.", "etc": "etc.",
             "vs": "vs.", "copyright": "\u00a9", "times": "\u00d7",
             "lsq": "\u2018", "rsq": "\u2019", "lq": "\u201c", "rq": "\u201d",
             "today": "", "timeofday": "", "midpoint": "\u00b7",
             "swungdash": "\u2053", "permille": "\u2030", "pilcrow": "\u00b6",
             "section-mark": "\u00a7", "minus": "\u2212", "solidus": "/",
             "obelus": "\u00f7", "plusminus": "\u00b1", "degree": "\u00b0",
             "prime": "\u2032", "dblprime": "\u2033", "icon": ""}


def _xml_load(path, seen=None):
    """Parse an XML file and read its xi:include files in (relative to it)."""
    import xml.etree.ElementTree as ET
    seen = seen or set()
    path = Path(path).resolve()
    seen.add(path)
    root = ET.parse(path).getroot()
    xi = "{http://www.w3.org/2001/XInclude}include"

    def expand(el):
        kids = list(el)
        for k, ch in enumerate(kids):
            if ch.tag == xi:
                f = (path.parent / ch.get("href", "")).resolve()
                if ch.get("parse") == "text":
                    new = ET.Element("pre")
                    new.text = f.read_text(encoding="utf-8") if f.is_file() \
                        else ""
                elif f.is_file() and f not in seen:
                    new = _xml_load(f, seen)
                else:
                    new = ET.Element("skipped")
                new.tail = ch.tail
                el.remove(ch)
                el.insert(k, new)
            else:
                expand(ch)
    expand(root)
    return root


PTX_IDS = {}            # xml:id -> label, filled by from_xml
PTX_NUM = {}            # id(element) -> its number, filled by from_xml
PTX_NUMBERED = {"theorem", "lemma", "proposition", "corollary", "definition",
                "example", "remark", "algorithm", "claim", "fact", "axiom",
                "conjecture", "note", "observation", "principle", "identity",
                "insight", "assumption", "heuristic", "hypothesis",
                "question", "problem", "activity", "exploration",
                "investigation", "project", "computation", "technology",
                "data", "convention", "warning", "figure", "table",
                "listing", "list"}


def _ptx_number(root):
    """Number blocks as PreTeXt does by default: one counter per chapter
    shared by theorems, examples, figures and the like (Example 3.8,
    Table 3.10); exercises in an exercises division count on their own.
    The numbers are right only when the whole book is deposited."""
    nums, state = {}, {"ch": 0, "blk": 0}

    def walk(el, in_ex):
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        if tag in PTX_SKIP:
            return
        if tag == "chapter":
            state["ch"] += 1
            state["blk"] = 0
            nums[id(el)] = str(state["ch"])
        elif tag in ("exercises", "reading-questions", "worksheet"):
            in_ex = [0]
        elif tag == "exercise" and in_ex:
            in_ex[0] += 1
            nums[id(el)] = str(in_ex[0])
        elif tag in PTX_NUMBERED or (tag == "exercise" and not in_ex):
            state["blk"] += 1
            nums[id(el)] = ("%d.%d" % (state["ch"], state["blk"])
                            if state["ch"] else str(state["blk"]))
        for ch in el:
            walk(ch, in_ex)
    walk(root, None)
    return nums


def _ptx_label(el):
    tag = el.tag.split("}")[-1]
    t = el.find("title")
    title = " ".join("".join(_ptx_inline(t)).split()) if t is not None else ""
    kind = tag.replace("-", " ").capitalize()
    num = PTX_NUM.get(id(el))
    if num:
        kind += " " + num
    xid = el.get("{http://www.w3.org/XML/1998/namespace}id")
    # an untitled, unnumbered block is named by its id, so a reference to
    # it can still be followed
    return "%s: %s" % (kind, title) if title else \
        (kind if num or not xid else "%s %s" % (kind, xid))


def _ptx_text(el, out, level_shift=0):
    tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
    if not tag:                                       # comment
        if el.tail:
            out.append(el.tail)
        return

    def kids():
        if el.text:
            out.append(el.text)
        for ch in el:
            _ptx_text(ch, out, level_shift)

    if tag in PTX_SKIP:
        pass
    elif tag in PTX_EMPTY and not len(el) and not (el.text or "").strip():
        out.append(PTX_EMPTY[tag])
    elif tag in PTX_DIVISIONS:
        t = el.find("title")
        title = "".join(_ptx_inline(t)) if t is not None else \
            tag.replace("-", " ").title()
        out.append(_heading(PTX_DIVISIONS[tag] + level_shift, title))
        if el.text:
            out.append(el.text)
        for ch in el:
            if ch is not t:
                _ptx_text(ch, out, level_shift)
    elif tag in ("title", "subtitle", "shorttitle", "plaintitle"):
        pass                                          # used by its parent
    elif tag == "m":
        out.append("$%s$" % (el.text or "").strip())
    elif tag in ("me", "men"):
        out.append("\n$$%s$$" % " ".join((el.text or "").split()))
    elif tag in ("md", "mdn"):
        rows = [" ".join((r.text or "").split()) for r in el.findall("mrow")]
        if len(rows) <= 1:
            out.append("\n$$%s$$" % (rows[0] if rows else
                                       " ".join((el.text or "").split())))
        else:
            body = " \\\\\n".join(rows) if rows else (el.text or "").strip()
            out.append("\n$$\\begin{aligned}\n%s\n\\end{aligned}$$" % body)
    elif tag == "c":
        out.append("`%s`" % (el.text or ""))
    elif tag == "q":
        out.append("\u201c")
        kids()
        out.append("\u201d")
    elif tag == "xref":
        ref = el.get("ref") or el.get("provisional") or ""
        out.append("[%s]" % ", ".join(PTX_IDS.get(r, "ref: " + r)
                                      for r in ref.replace(",", " ").split()))
    elif tag in ("url", "dataurl"):
        out.append(el.text or el.get("href") or "")
    elif tag == "li":
        out.append("\n- ")
        kids()
    elif tag in ("figure", "table", "listing", "list"):
        out.append("\n\n%s.\n" % _ptx_label(el))
        kids()
        out.append("\n\n")
    elif tag == "cell":
        kids()
        out.append(" | ")
    elif tag == "row":
        out.append("\n| ")
        kids()
        out.append("\n")
    elif tag in ("p", "ol", "ul", "dl", "statement", "introduction",
                 "conclusion", "blockquote", "pre", "cd", "tabular",
                 "biblio", "task", "sidebyside", "caption"):
        out.append("\n\n")
        kids()
        out.append("\n\n")
    elif tag in PTX_BLOCKS:
        t = el.find("title")
        label = _ptx_label(el)
        out.append("\n\n%s.\n" % label)
        if el.text:
            out.append(el.text)
        for ch in el:
            if ch is not t:
                _ptx_text(ch, out, level_shift)
        out.append("\n\n")
    else:
        kids()
    if el.tail:
        out.append(el.tail)


def _ptx_inline(el):
    out = []
    if el.text:
        out.append(el.text)
    for ch in el:
        _ptx_text(ch, out)
    return out


def from_xml(text, path=None):
    """XML. PreTeXt (and its older name, MathBook XML): chapters, sections and
    subsections become headings, xi:include files are read in, <m>, <me> and
    <md> become LaTeX, theorems, definitions and examples get a label, and
    Sage cells, index entries and pictures are left out. Other XML: its text."""
    import xml.etree.ElementTree as ET
    root = _xml_load(path) if path else ET.fromstring(text)
    out = []
    names = {e.tag.split("}")[-1] for e in root.iter() if isinstance(e.tag, str)}
    if names & {"pretext", "mathbook", "chapter", "section", "subsection"} \
            or "m" in names:
        xid = "{http://www.w3.org/XML/1998/namespace}id"
        PTX_NUM.clear()
        PTX_NUM.update(_ptx_number(root))
        PTX_IDS.clear()
        PTX_IDS.update((e.get(xid), _ptx_label(e)) for e in root.iter()
                       if isinstance(e.tag, str) and e.get(xid))
        _ptx_text(root, out)
        raw = "".join(out)
    else:
        raw = "\n\n".join(" ".join(t.split()) for t in root.itertext()
                          if t.strip())
    return _tidy(_guard_hashes_body(_unwrap(raw)))


def _unwrap(raw):
    """Join the lines of each paragraph; keep headings, '- ' items and
    displayed mathematics on lines of their own."""
    lines = [l if re.match(r"#{1,6} \S", l) else " ".join(l.split())
             for l in raw.split("\n")]
    text = re.sub(r"\n-\n+", "\n- ", "\n".join(lines))
    paras = []
    for para in re.split(r"\n\s*\n", text):
        out, in_math = [], False
        for l in para.split("\n"):
            if not l:
                continue
            starts = l.startswith("$$")
            if in_math or starts or l.startswith(("- ", "#")) or not out \
                    or out[-1].endswith("$$") and not l[:1] in ".,;:)":
                out.append(l)
            else:
                out[-1] += ("" if l[:1] in ".,;:)" else " ") + l
            if l.count("$$") % 2:
                in_math = not in_math
        paras.append("\n".join(out))
    return "\n\n".join(p for p in paras if p)


SOURCE_FORMATS = {".txt": None, ".text": None, ".md": from_markdown,
                  ".markdown": from_markdown, ".tex": from_latex,
                  ".latex": from_latex, ".html": from_html, ".htm": from_html,
                  ".xhtml": from_html, ".xml": from_xml, ".ptx": from_xml}


def source_to_text(path):
    """Read a book for deposit: (text, format name)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".pdf", ".docx", ".doc", ".epub", ".odt", ".rtf"):
        die("%s is not a source format. Look for the book's source first "
            "(LaTeX, PreTeXt XML, HTML or Markdown: arXiv and open textbooks "
            "often publish it); otherwise turn it into Markdown with: "
            "ingest %s --deposit" % (ext, path.name))
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        die("not a UTF-8 text file")
    conv = SOURCE_FORMATS.get(ext)
    if conv is None:
        return text, "text"
    try:
        return conv(text, path), ext.lstrip(".")
    except Exception as e:          # malformed XML and the like
        die("could not read %s as %s: %s" % (path.name, ext, e))


def select_parts(text, parts):
    """Keep only the top-level ('# ') divisions named in parts, by their
    position (1, 2, ...) or their title (case-insensitive)."""
    pieces, cur = [], None
    for line in text.split("\n"):
        if line.startswith("# "):
            cur = [line[2:].strip(), [line]]
            pieces.append(cur)
        elif cur:
            cur[1].append(line)
    if not pieces:
        die("--part needs '#' headings, and this book has none")
    keep, missing = [], []
    for want in parts:
        hit = [k for k, (title, _) in enumerate(pieces, 1)
               if want.strip() == str(k)
               or want.strip().casefold() == title.casefold()]
        if not hit:
            missing.append(want)
        keep += [k for k in hit if k not in keep]
    if missing:
        die("no top-level division %s; the book has: %s" % (
            ", ".join(repr(m) for m in missing), "; ".join(
                "%d %s" % (k, t) for k, (t, _) in enumerate(pieces, 1))))
    return "\n".join("\n".join(pieces[k - 1][1]) for k in sorted(keep)), \
        [pieces[k - 1][0] for k in sorted(keep)]


def cmd_deposit(a):
    """Store a book in the library as text with '#' headings, converted
    from its source format; identical content is refused."""
    st = Store(a.project)
    src = Path(a.file)
    if not src.is_file():
        die("no such file: %s" % src)
    text, fmt = source_to_text(src)
    name = src.name
    if getattr(a, "part", None):
        text, titles = select_parts(text, a.part)
        name = "%s [%s]" % (src.name, "; ".join(titles))
    if not text.strip():
        die("the file is empty")
    sha = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    ldir, idx, books = _library(st)
    for b, info in books.items():
        if info["sha256"] == sha:
            die("same content already deposited as %s (%s)" % (b, info["name"]))
    bid = sha[:8]
    (ldir / (bid + ".txt")).write_text(text, encoding="utf-8")
    books[bid] = {"name": name, "sha256": sha, "chars": len(text),
                  "format": fmt, "date": time.strftime("%Y-%m-%d")}
    idx.write_text(json.dumps(books, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    heads = sum(1 for l in text.split("\n") if re.match(r"#{1,6} \S", l))
    print("deposited %s  %s  (%s, %d characters, %d headings)"
          % (bid, name, fmt, len(text), heads))


def cmd_library(a):
    st = Store(a.project)
    ldir, _, books = _library(st)
    for b, info in sorted(books.items(), key=lambda x: x[1]["date"]):
        rep = ldir / (b + ".filter.json")
        state = "deposited"
        if rep.is_file():
            done = json.loads(rep.read_text(encoding="utf-8")).get("applied")
            state = "accepted" if done else "filtered, awaiting accept"
        print("%s  %s  %8d chars  %s  [%s]" % (b, info["date"], info["chars"],
                                               info["name"], state))


def split_chunks(text, name, size):
    lines = text.splitlines()
    chunks = []
    if any(l.startswith("#") for l in lines):
        title, body, stack = name, [], []
        for line in lines:
            if line.startswith("#"):
                if "\n".join(body).strip():
                    chunks.append((title, "\n".join(body).strip()))
                depth = len(line) - len(line.lstrip("#"))
                stack = [h for h in stack if h[0] < depth]
                stack.append((depth, line.lstrip("#").strip() or name))
                # the headings above it too: two sections may share a title
                title, body = " > ".join(h[1] for h in stack), []
            else:
                body.append(line)
        if "\n".join(body).strip():
            chunks.append((title, "\n".join(body).strip()))
        return chunks
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if cur and len(cur) + len(para) > size:
            chunks.append(("%s (%d)" % (name, len(chunks) + 1), cur))
            cur = ""
        cur = (cur + "\n\n" + para.strip()).strip()
    if cur:
        chunks.append(("%s (%d)" % (name, len(chunks) + 1), cur))
    return chunks


FILTER_HEAD = (
    "You are filtering a deposited book for a personal knowledge tree. Decide "
    "whether the passage below belongs on the reading side of one of the "
    "listed files. Keep it only if it carries real subject content about "
    "that file's subject, and choose the most specific file. Reject "
    "publisher matter, acknowledgements, advertisements, tables of contents "
    "and anything off-topic.%s\nReply with JSON only: {\"keep\": true or "
    "false, \"target\": \"id of the file it belongs to, or empty\", "
    "\"title\": \"short title for the passage\", \"reason\": \"one "
    "sentence\"}")


def cmd_filter(a):
    """Ask the model, passage by passage, what to keep and where it belongs.
    Writes a review report only; the tree is not touched until 'accept'."""
    st = Store(a.project)
    ldir, bid, info = _book(st, a.book)
    text = (ldir / (bid + ".txt")).read_text(encoding="utf-8")
    parent = st.resolve(a.parent) if a.parent else ""
    rows = [st.nodes[nid] for nid in st.subtree_ids(parent)]
    if not rows:
        die("the tree is empty; create files first")
    ok_targets = {n.id for n in rows}
    table = "\n".join("%s | %s | %s | %s" % (n.id, st.number(n.id),
                       LEVELS[n.level - 1], n.title) for n in rows)
    use_model(a)
    head = FILTER_HEAD % ((" Extra criteria: " + a.ask) if a.ask else "")
    chunks = split_chunks(text, Path(info["name"]).stem, a.size)

    def prompt_for(title, body):
        return ("%s\n\n# Tree files (id | number | level | title)\n%s\n\n"
                "# Passage: %s\n%s" % (head, table, title, body))
    if a.backend == "dry":
        print(prompt_for(*chunks[0]))
        print("\n[dry run: %d passages = %d model calls, %d characters in the "
              "book]" % (len(chunks), len(chunks), len(text)), file=sys.stderr)
        return
    items = []
    for i, (title, body) in enumerate(chunks, 1):
        try:
            ans = call_model(st.root, a.backend, prompt_for(title, body))[0]
        except Exception as e:
            die("model call failed at passage %d: %s: %s (nothing was written)"
                % (i, type(e).__name__, e))
        item = {"n": i, "status": "ok", "keep": False, "target": "",
                "title": title, "reason": "", "chars": len(body), "text": body}
        try:
            d = json.loads(re.search(r"\{.*\}", ans, re.S).group(0))
            item["keep"] = d.get("keep") is True
            item["target"] = str(d.get("target") or "")
            item["title"] = str(d.get("title") or title)
            item["reason"] = str(d.get("reason") or "")
            if item["keep"] and item["target"] not in ok_targets:
                item.update(status="review", keep=False,
                            reason="model chose an invalid target '%s'. %s"
                                   % (item["target"], item["reason"]))
        except Exception:
            item.update(status="review", reason="unreadable model reply: "
                                                + ans.strip()[:200])
        items.append(item)
        if item["status"] == "review":
            verdict = "REVIEW"
        elif item["keep"]:
            verdict = "KEEP -> " + st.number(item["target"])
        else:
            verdict = "DROP"
        print("%3d  %-16s %s  | %s" % (i, verdict, item["title"], item["reason"]))
    rep = ldir / (bid + ".filter.json")
    rep.write_text(json.dumps({"book": bid, "name": info["name"],
                               "applied": False, "items": items},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    kept = sum(1 for x in items if x["keep"])
    review = sum(1 for x in items if x["status"] == "review")
    print("%d kept, %d dropped, %d need review. Report: %s"
          % (kept, len(items) - kept - review, review, rep))
    print("Edit the report if you disagree, then run: accept %s" % bid)


def cmd_accept(a):
    """Append the kept passages of a filter report to the reading side of
    their target files."""
    st = Store(a.project)
    ldir, bid, _ = _book(st, a.book)
    rep = ldir / (bid + ".filter.json")
    if not rep.is_file():
        die("this book has not been filtered yet")
    data = json.loads(rep.read_text(encoding="utf-8"))
    if data.get("applied"):
        die("this report was already accepted")
    made = skipped = 0
    backed_up = set()
    for item in data["items"]:
        if item.get("keep") is not True:
            continue
        t = st.nodes.get(item.get("target", ""))
        if t is None:
            print("skipped passage %d: target '%s' does not exist"
                  % (item["n"], item.get("target", "")))
            skipped += 1
            continue
        marks = {MARK.format(sec) for sec in SECTIONS}
        body = "\n".join(("> " + l) if l.strip() in marks else l
                         for l in item["text"].splitlines())
        block = "## %s\n[source: %s, book %s, passage %d]\n%s" % (
            item["title"], data["name"], bid, item["n"], body)
        if t.id not in backed_up:
            _backup(st, t)
            backed_up.add(t.id)
        t.reading = (t.reading.strip("\n") + "\n\n" + block).strip("\n")
        st.save(t)
        made += 1
        print("%s  [%s]  %s  <-  %s" % (st.number(t.id), t.id, t.title,
                                        item["title"]))
    data["applied"] = True
    rep.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("%d passage(s) added to reading sides, %d skipped" % (made, skipped))


def cmd_read(a):
    """Print the reading side of a file, or of a file and its whole subtree."""
    st = Store(a.project)
    nid = st.resolve(a.ref)

    def emit(n):
        if n.reading.strip():
            print("# %s  %s [%s]\n\n%s\n" % (st.number(n.id), n.title, n.id,
                                              n.reading))
            return 1
        return 0

    def walk(pid):
        return sum(emit(k) + walk(k.id) for k in st.children(pid))
    count = emit(st.nodes[nid]) + (walk(nid) if a.all else 0)
    if not count:
        print("nothing to read here yet (deposit a book, then filter and accept)")


__all__ = [
    'FILTER_HEAD', 'PTX_BLOCKS', 'PTX_DIVISIONS', 'PTX_EMPTY', 'PTX_IDS',
    'PTX_NUM', 'PTX_NUMBERED', 'PTX_SKIP', 'SOURCE_FORMATS',
    'TEX_ACCENTS', 'TEX_BLOCKS', 'TEX_DROP', 'TEX_LAYOUT', 'TEX_LEVELS',
    'TEX_MATH', 'TEX_NOTTEXT', 'TEX_STYLE', 'TEX_VERBATIM', 'TEX_WORDS',
    '_HTMLText', '_book', '_guard_hashes_body', '_heading', '_library',
    '_ptx_inline', '_ptx_label', '_ptx_number', '_ptx_text', '_tex_arg',
    '_tex_expand', '_tex_includes', '_tex_macros', '_tex_strip_comments',
    '_tidy', '_unwrap', '_xml_load', 'cmd_accept', 'cmd_deposit',
    'cmd_filter', 'cmd_library', 'cmd_read', 'from_html', 'from_latex',
    'from_markdown', 'from_xml', 'select_parts', 'source_to_text',
    'split_chunks',
]
