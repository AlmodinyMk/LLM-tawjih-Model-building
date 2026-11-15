"""Tkinter front end: the same four windows, standard library only.

Note: Tk on Linux does not shape or reorder Arabic text, so Arabic content
is shown with unjoined letters. Use the GTK front end for Arabic content.
"""
import keyword
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from desk_core import (AI_BACKENDS, CHAT_CONTEXT_LIMIT, SIDES, STATUSES, Desk,
                       DeskError, ai_busy)

WAITING = "Waiting for another AI request of this program to finish …"

MONO = ("DejaVu Sans Mono", 10)
TOKEN = re.compile(
    r"(?P<com>#[^\n]*)"
    r"|(?P<str>(\"\"\"|''')[\s\S]*?(?:\3|\Z)|\"(?:\\.|[^\"\\\n])*\"?"
    r"|'(?:\\.|[^'\\\n])*'?)"
    r"|(?P<num>\b\d+(?:\.\d+)?\b)"
    r"|(?P<kw>\b(?:%s)\b)" % "|".join(keyword.kwlist))
COLORS = {"com": "#6a737d", "str": "#0a7d33", "num": "#b5450b",
          "kw": "#7c2fb8"}


# ------------------------------------------------------------------ helpers
def get_text(t):
    return t.get("1.0", "end-1c")


def set_text(t, text):
    state = str(t.cget("state"))
    t.configure(state="normal")
    t.delete("1.0", "end")
    t.insert("1.0", text)
    t.edit_reset()      # undo must never bring back the previous file
    t.configure(state=state)
    t.event_generate("<<Refresh>>")


def selection_or_all(t):
    try:
        return t.get("sel.first", "sel.last")
    except tk.TclError:
        return get_text(t)


def text_box(parent, editable=True, mono=False, height=8, wrap="word"):
    frame = ttk.Frame(parent)
    t = tk.Text(frame, wrap=wrap, height=height, undo=True,
                font=MONO if mono else None, padx=6, pady=6)
    bar = ttk.Scrollbar(frame, command=t.yview)
    t.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    t.pack(side="left", fill="both", expand=True)
    if not editable:
        t.configure(state="disabled")
    return frame, t


def table(parent, columns, height=6):
    """A sortable table. columns: [(key, title, width)]."""
    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=[c[0] for c in columns],
                        show="headings", height=height, selectmode="browse")
    bar = ttk.Scrollbar(frame, command=tree.yview)
    tree.configure(yscrollcommand=bar.set)

    def sort(col, reverse=False):
        def key(item):
            v = tree.set(item, col)
            return (0, int(v)) if v.isdigit() else (1, v.casefold())
        for i, item in enumerate(sorted(tree.get_children(""), key=key,
                                        reverse=reverse)):
            tree.move(item, "", i)
        tree.heading(col, command=lambda: sort(col, not reverse))
    for key, title, width in columns:
        tree.heading(key, text=title, command=lambda c=key: sort(c))
        tree.column(key, width=width, anchor="w")
    bar.pack(side="right", fill="y")
    tree.pack(side="left", fill="both", expand=True)
    return frame, tree


def fill(tree, rows):
    tree.delete(*tree.get_children(""))
    for iid, values in rows:
        tree.insert("", "end", iid=iid, values=values)


def selected(tree):
    sel = tree.selection()
    return sel[0] if sel else None


def bar(parent, *widgets):
    for w in widgets:
        w.pack(in_=parent, side="left", padx=3, pady=2,
               fill="x" if type(w) in (ttk.Entry, Hint) else "none",
               expand=type(w) in (ttk.Entry, Hint))
    return parent


class Hint(ttk.Entry):
    """An entry with a label in front of it (Tk has no placeholder)."""

    def __init__(self, parent, label, width=14):
        self.frame = ttk.Frame(parent)
        ttk.Label(self.frame, text=label).pack(side="left", padx=(0, 3))
        super().__init__(self.frame, width=width)
        super().pack(side="left", fill="x", expand=True)

    def pack(self, **kw):
        kw.pop("in_", None)
        self.frame.pack(**kw)

    def set(self, text):
        self.delete(0, "end")
        self.insert(0, text)


class CodeText(ttk.Frame):
    """Code editor: line numbers, colours, auto-indent, 4-space tabs."""

    def __init__(self, parent):
        super().__init__(parent)
        self.gutter = tk.Text(self, width=4, font=MONO, state="disabled",
                              bg="#eeeeee", fg="#777777", bd=0, pady=6,
                              takefocus=0, height=6)
        self.text = tk.Text(self, wrap="none", undo=True, font=MONO, padx=6,
                            pady=6, height=6)
        yb = ttk.Scrollbar(self, command=self._yview)
        self.text.configure(yscrollcommand=lambda a, b: (
            yb.set(a, b), self.gutter.yview_moveto(a)))
        self.gutter.pack(side="left", fill="y")
        yb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        for tag, color in COLORS.items():
            self.text.tag_configure(tag, foreground=color)
        self.text.tag_configure("err", background="#ffd6d6")
        self.text.bind("<Return>", self._newline)
        self.text.bind("<Tab>", self._tab)
        self.text.bind("<BackSpace>", self._backspace)
        self.text.bind("<KeyRelease>", lambda e: self.refresh())
        self.text.bind("<<Refresh>>", lambda e: self.refresh())
        self.refresh()

    def _yview(self, *args):
        self.text.yview(*args)
        self.gutter.yview(*args)

    def _newline(self, event):
        line = self.text.get("insert linestart", "insert")
        indent = re.match(r"[ \t]*", line).group(0)
        if line.rstrip().endswith(":"):
            indent += "    "
        self.text.insert("insert", "\n" + indent)
        self.text.see("insert")
        return "break"

    def _tab(self, event):
        self.text.insert("insert", "    ")
        return "break"

    def _backspace(self, event):
        before = self.text.get("insert linestart", "insert")
        if before and not before.strip() and len(before) % 4 == 0:
            self.text.delete("insert-4c", "insert")
            return "break"
        return None

    def refresh(self):
        code = get_text(self.text)
        for tag in COLORS:
            self.text.tag_remove(tag, "1.0", "end")
        for m in TOKEN.finditer(code):
            self.text.tag_add(m.lastgroup, "1.0+%dc" % m.start(),
                              "1.0+%dc" % m.end())
        lines = code.count("\n") + 1
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "\n".join("%3d" % i
                                            for i in range(1, lines + 1)))
        self.gutter.configure(state="disabled")
        self.gutter.yview_moveto(self.text.yview()[0])

    def mark_error(self, line=None):
        self.text.tag_remove("err", "1.0", "end")
        if line:
            self.text.tag_add("err", "%d.0" % line, "%d.0 lineend+1c" % line)
            self.text.mark_set("insert", "%d.0" % line)
            self.text.see("%d.0" % line)


# -------------------------------------------------------------- base window
class DeskWindow(tk.Toplevel):
    TITLE, SIZE = "Tawjih", "900x600"

    def __init__(self, app):
        super().__init__(app.root)
        self.app, self.desk = app, Desk(app.project)
        self.title("Tawjih · " + self.TITLE)
        self.geometry(self.SIZE)
        self.ai_buttons, self.busy = [], False
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 0))
        self.ai_on = tk.BooleanVar(value=False)
        self.ai_backend = ttk.Combobox(top, values=AI_BACKENDS, width=10,
                                       state="readonly")
        self.ai_backend.current(0)
        self.ai_model = ttk.Entry(top, width=22)
        self.ai_model.insert(0, os.environ.get("TAWJIH_MODEL", ""))
        self.chat = None
        bar(top, ttk.Checkbutton(top, text="Use AI", variable=self.ai_on,
                                 command=self._ai_state),
            ttk.Label(top, text="provider"), self.ai_backend,
            ttk.Label(top, text="model"), self.ai_model,
            self.button(top, "Ask AI…", self.open_chat, True))
        self.status = ttk.Label(self, anchor="w", wraplength=880)
        self.status.pack(side="bottom", fill="x", padx=8, pady=6)
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True, padx=8, pady=6)
        self.build()
        self._ai_state()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def build(self):
        raise NotImplementedError

    def closing(self):
        """Return True to keep the window open."""
        return False

    def open_chat(self):
        """The side conversation of this window (one per window)."""
        if self.chat is None:
            self.chat = ChatWindow(self)
        self.chat.lift()
        return self.chat

    def chat_context(self):
        """What this window currently shows, as text for the conversation."""
        return ""

    def close(self):
        if self.closing():
            return
        if self.chat is not None:
            self.chat.close()
        self.destroy()
        self.app.closed(self)

    def say(self, text):
        self.status.configure(text=text)

    def guard(self, fn, *args):
        try:
            res = fn(*args)
        except DeskError as e:
            self.say("Error: %s" % e)
            return None
        return res if res is not None else True

    def confirm(self, text):
        return messagebox.askyesno("Tawjih", text, parent=self)

    def button(self, parent, label, handler, ai=False):
        b = ttk.Button(parent, text=label, command=handler)
        if ai:
            self.ai_buttons.append(b)
        return b

    def _ai_state(self):
        on = self.ai_on.get()
        self.ai_backend.configure(state="readonly" if on else "disabled")
        self.ai_model.configure(state="normal" if on else "disabled")
        for b in self.ai_buttons:
            b.configure(state="normal" if on and not self.busy else "disabled")
        if self.chat is not None:
            self.chat.refresh_state()

    def ai_backend_name(self):
        return self.ai_backend.get()

    def ai_model_name(self):
        return self.ai_model.get().strip()

    def run_async(self, label, work, done):
        self.say(WAITING if ai_busy() else label + " …")
        self.busy = True
        self._ai_state()

        def finish(result, error):
            self.busy = False
            if not self.winfo_exists():
                return
            self._ai_state()
            if error:
                self.say("Error: %s" % error)
            else:
                self.say(label + ": done")
                done(result)

        def target():
            try:
                res, err = work(), None
            except DeskError as e:
                res, err = None, str(e)
            except Exception as e:      # never lose an error in a thread
                res, err = None, "%s: %s" % (type(e).__name__, e)
            self.app.results.put((finish, res, err))
        threading.Thread(target=target, daemon=True).start()

    def run_ai(self, label, action, text, done, **kw):
        if not self.ai_on.get():
            return
        backend = self.ai_backend_name()
        model = self.ai_model_name()
        self.run_async(label, lambda: self.desk.ai(action, text, backend,
                                                   model=model, **kw), done)


class ChatWindow(tk.Toplevel):
    """A side conversation with the provider, next to one main window.
    It has no way to change the project: there is nothing to apply and
    nothing is saved, not even the conversation."""

    def __init__(self, owner):
        super().__init__(owner.app.root)
        self.owner, self.history, self.busy = owner, [], False
        self.title("Tawjih · Ask AI · " + owner.TITLE)
        self.geometry("520x600")
        ttk.Label(self, wraplength=500, text="A side conversation. Nothing "
                  "here is saved, and nothing here changes the project."
                  ).pack(anchor="w", padx=8, pady=(8, 4))
        self.status = ttk.Label(self, anchor="w", wraplength=500)
        self.status.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        row = ttk.Frame(self)
        row.pack(side="bottom", fill="x", padx=8, pady=4)
        self.send_btn = ttk.Button(row, text="Send  (Enter)", command=self.ask)
        bar(row, self.send_btn, ttk.Button(row, text="Clear conversation",
                                           command=self.reset))
        frame, self.entry = text_box(self, height=4)
        frame.pack(side="bottom", fill="x", padx=8)
        self.entry.bind("<Return>", self._enter)
        self.include = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, variable=self.include, text="Send what is open "
                        "in the main window with my question").pack(
            side="bottom", anchor="w", padx=8, pady=4)
        frame, self.view = text_box(self, editable=False, height=10)
        frame.pack(fill="both", expand=True, padx=8)
        self.view.tag_configure("who", font=("TkDefaultFont", 10, "bold"))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_state()

    def close(self):
        if self.owner.chat is self:
            self.owner.chat = None
        self.destroy()

    def _enter(self, event):
        if event.state & 0x1:           # Shift+Enter: a new line
            return None
        self.ask()
        return "break"

    def refresh_state(self):
        on = self.owner.ai_on.get()
        self.send_btn.configure(
            state="normal" if on and not self.busy else "disabled")
        if not on:
            self.status.configure(text="AI is off: tick \"Use AI\" in the "
                                  "main window to ask.")
        elif not self.busy:
            self.status.configure(text="")

    def add_line(self, who, text):
        self.view.configure(state="normal")
        self.view.insert("end", who + "\n", "who")
        self.view.insert("end", text + "\n\n")
        self.view.configure(state="disabled")
        self.view.see("end")

    def reset(self):
        self.history = []
        set_text(self.view, "")

    def ask(self):
        question = get_text(self.entry).strip()
        if self.busy or not question or not self.owner.ai_on.get():
            return
        context = self.owner.chat_context() if self.include.get() else ""
        backend, model = self.owner.ai_backend_name(), \
            self.owner.ai_model_name()
        history, desk, title = list(self.history), self.owner.desk, \
            self.owner.TITLE
        set_text(self.entry, "")
        self.add_line("You", question)
        self.busy = True
        self.refresh_state()
        self.status.configure(text=WAITING if ai_busy() else "Asking …")

        def finish(result, error):
            if not self.winfo_exists():
                return
            self.busy = False
            self.refresh_state()
            if error:
                self.status.configure(text="Error: %s" % error)
                self.add_line("No answer", error)
                if not get_text(self.entry).strip():
                    set_text(self.entry, question)   # nothing to retype
                return
            answer, cut = result
            self.history += [("User", question), ("Assistant", answer)]
            self.add_line("AI", answer)
            self.status.configure(
                text="Only the first %d characters of the window's content "
                "were sent." % CHAT_CONTEXT_LIMIT if cut else "")

        def target():
            try:
                res, err = desk.chat(question, backend, history, title,
                                     context, model), None
            except DeskError as e:
                res, err = None, str(e)
            except Exception as e:      # never lose an error in a thread
                res, err = None, "%s: %s" % (type(e).__name__, e)
            self.owner.app.results.put((finish, res, err))
        threading.Thread(target=target, daemon=True).start()


class Proposal(ttk.Frame):
    """Shows a model's answer. Nothing changes until the user applies it."""

    def __init__(self, parent):
        super().__init__(parent)
        self.label = ttk.Label(self, font=("TkDefaultFont", 10, "bold"))
        self.label.pack(anchor="w")
        frame, self.view = text_box(self, height=4)
        frame.pack(fill="both", expand=True)
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.apply_btn = ttk.Button(row, text="Apply", command=self._apply)
        bar(row, self.apply_btn,
            ttk.Button(row, text="Discard", command=self.clear))
        self._cb, self.visible = None, False

    def show_text(self, title, text, apply_cb=None):
        self.label.configure(text=title)
        set_text(self.view, text)
        self._cb = apply_cb
        self.apply_btn.configure(state="normal" if apply_cb else "disabled")
        if not self.visible:
            self.pack(fill="both", expand=True, pady=(6, 0))
            self.visible = True

    def _apply(self):
        if self._cb:
            self._cb(get_text(self.view))
            self.clear()

    def clear(self):
        self._cb = None
        if self.visible:
            self.pack_forget()
            self.visible = False


class FileList(ttk.Frame):
    def __init__(self, parent, on_select):
        super().__init__(parent)
        self.tree = ttk.Treeview(self, columns=("title", "sides"),
                                 selectmode="browse", height=12)
        self.tree.heading("#0", text="No.")
        self.tree.heading("title", text="Title")
        self.tree.heading("sides", text="Sides")
        self.tree.column("#0", width=120)
        self.tree.column("title", width=170)
        self.tree.column("sides", width=50)
        sb = ttk.Scrollbar(self, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self._filling, self._on_select = False, on_select
        self.tree.bind("<<TreeviewSelect>>", self._changed)

    def fill(self, rows, keep=None):
        self._filling = True
        self.tree.delete(*self.tree.get_children(""))
        for r in rows:
            self.tree.insert(r["parent"], "end", iid=r["id"], open=True,
                             text=r["number"], values=(r["title"], r["flags"]))
        if keep and self.tree.exists(keep):
            self.tree.selection_set(keep)
        self.update_idletasks()
        self._filling = False

    def select(self, nid):
        self._filling = True
        if nid and self.tree.exists(nid):
            self.tree.selection_set(nid)
        self.update_idletasks()
        self._filling = False

    def _changed(self, event):
        if not self._filling and self.tree.selection():
            self._on_select(self.tree.selection()[0])


# ------------------------------------------------------- window 1: library
class LibraryWindow(DeskWindow):
    TITLE, SIZE = "Books and research", "1000x680"

    def build(self):
        self.book = None
        top = ttk.Frame(self.body)
        top.pack(fill="x")
        self.query = Hint(top, "search books", 30)
        self.query.bind("<Return>", lambda e: self.search())
        bar(top, self.button(top, "Deposit…", self.deposit),
            self.button(top, "Refresh", self.refresh), self.query,
            self.button(top, "Search", self.search))
        panes = ttk.Panedwindow(self.body, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=6)
        left = ttk.Panedwindow(panes, orient="vertical")
        f, self.books = table(left, [("id", "Id", 80), ("name", "Name", 130),
                                     ("date", "Date", 90),
                                     ("chars", "Characters", 80),
                                     ("state", "State", 80)])
        left.add(f, weight=1)
        f, self.hits = table(left, [("name", "Name", 110), ("line", "Line", 50),
                                    ("text", "Text", 280)])
        left.add(f, weight=1)
        self.books.bind("<<TreeviewSelect>>", lambda e: self._book_changed())
        self.hits.bind("<Double-1>", lambda e: self.open_hit())
        self.hit_rows = {}
        right = ttk.Frame(panes)
        f, self.text = text_box(right, editable=False, height=6)
        f.pack(fill="both", expand=True)
        r1 = ttk.Frame(right)
        r1.pack(fill="x")
        self.lang = Hint(r1, "to", 14)
        bar(r1, self.button(r1, "Translate selection", self.translate, True),
            self.lang)
        r2 = ttk.Frame(right)
        r2.pack(fill="x")
        self.criteria = Hint(r2, "criteria", 18)
        bar(r2, self.button(r2, "Filter into the trees", self.filter_book,
                            True), self.criteria,
            self.button(r2, "Accept filter report", self.accept))
        self.out = Proposal(right)
        panes.add(left, weight=1)
        panes.add(right, weight=2)
        self.refresh()

    def chat_context(self):
        if not self.book:
            return ""
        return "Deposited book %s:\n%s" % (self.book,
                                           selection_or_all(self.text))

    def refresh(self):
        rows = self.guard(self.desk.books)
        rows = [] if rows in (None, True) else rows
        fill(self.books, [(b["id"], (b["id"], b["name"], b["date"],
                                     b["chars"], b["state"])) for b in rows])
        self.say("%d book(s) in the library" % len(rows))

    def deposit(self):
        path = filedialog.askopenfilename(
            parent=self, title="Deposit a text book or paper")
        if path:
            self.deposit_path(path)

    def deposit_path(self, path):
        msg = self.guard(self.desk.deposit, path)
        if msg:
            self.refresh()
            self.say(str(msg))

    def _book_changed(self):
        bid = selected(self.books)
        if bid:
            self.load(bid)

    def load(self, bid, line=None):
        text = self.guard(self.desk.book_text, bid)
        if text is None:
            return
        self.book = bid
        set_text(self.text, "" if text is True else text)
        if line:
            self.text.see("%d.0" % line)
            self.text.tag_remove("sel", "1.0", "end")
            self.text.tag_add("sel", "%d.0" % line, "%d.0 lineend" % line)

    def search(self):
        rows = self.guard(self.desk.search_books, self.query.get())
        rows = [] if rows in (None, True) else rows
        self.hit_rows = {str(i): h for i, h in enumerate(rows)}
        fill(self.hits, [(k, (h["name"], h["line"], h["text"]))
                         for k, h in self.hit_rows.items()])
        self.say("%d line(s) found; double-click one to open it" % len(rows))

    def open_hit(self, key=None):
        key = key or selected(self.hits)
        if key in self.hit_rows:
            self.load(self.hit_rows[key]["id"], self.hit_rows[key]["line"])

    def translate(self):
        if not self.text.tag_ranges("sel"):
            return self.say("Select the passage to translate first")
        self.run_ai("Translate", "translate", selection_or_all(self.text),
                    lambda t: self.out.show_text("Translation", t),
                    lang=self.lang.get())

    def filter_book(self):
        if not self.book:
            return self.say("Select a book first")
        if not self.ai_on.get():
            return None
        backend, bid, ask = self.ai_backend_name(), self.book, \
            self.criteria.get()
        model = self.ai_model_name()
        count = self.guard(self.desk.passage_count, bid)
        if count in (None, True) or not self.confirm(
                "Filtering sends this book to %s in %d passage(s), one call "
                "each. Continue?" % (backend, count)):
            return None

        def done(report):
            self.refresh()
            self.out.show_text("Filter report (nothing added yet; review, "
                               "then Accept)", report)
        self.run_async("Filter",
                       lambda: self.desk.filter_book(bid, backend, ask, model),
                       done)

    def accept(self):
        if not self.book:
            return self.say("Select a book first")
        msg = self.guard(self.desk.accept_book, self.book)
        if msg:
            self.refresh()
            self.out.show_text("Accepted into reading sides", str(msg))


# -------------------------------------------------- editors: windows 2 and 3
class EditorWindow(DeskWindow):
    EDIT_SIDES = ()

    def build(self):
        self.nid, self.loaded, self.editors = None, {}, {}
        panes = ttk.Panedwindow(self.body, orient="horizontal")
        panes.pack(fill="both", expand=True)
        self.files = FileList(panes, self.select)
        self.right = ttk.Frame(panes)
        self.heading = ttk.Label(self.right,
                                 font=("TkDefaultFont", 10, "bold"))
        self.heading.pack(anchor="w")
        self.build_editors()
        self.proposal = Proposal(self.right)
        panes.add(self.files, weight=1)
        panes.add(self.right, weight=3)
        self.refresh()

    def build_editors(self):
        raise NotImplementedError

    def refresh(self):
        rows = self.guard(self.desk.tree_rows)
        self.files.fill([] if rows in (None, True) else rows, keep=self.nid)

    def chat_context(self):
        if self.nid is None:
            return ""
        n = self.desk.st.nodes.get(self.nid)
        head = "File %s: %s" % (self.nid, n.title if n else "")
        return head + "".join("\n\n## %s side\n%s" % (
            s, get_text(self.editors[s])) for s in self.EDIT_SIDES)

    def dirty(self):
        return self.nid is not None and any(
            get_text(self.editors[s]).strip("\n") != self.loaded.get(s, "")
            for s in self.EDIT_SIDES)

    def leave_ok(self):
        if self.dirty() and self.confirm("Save your changes first?"):
            return self.save()
        return True

    def closing(self):
        return not self.leave_ok()      # a failed save keeps the window open

    def select(self, nid):
        if nid == self.nid:
            return
        if not self.leave_ok():
            self.files.select(self.nid)
            return
        self.load(nid)

    def load(self, nid):
        texts = {}
        for s in self.EDIT_SIDES:
            t = self.guard(self.desk.side, nid, s)
            if t is None:
                return
            texts[s] = "" if t is True else t
        self.nid, self.loaded = nid, texts
        for s, t in texts.items():
            set_text(self.editors[s], t)
        self.heading.configure(text="%s  %s" % (
            self.desk.st.number(nid), self.desk.st.nodes[nid].title))
        self.proposal.clear()
        self.say("")

    def save(self):
        if self.nid is None:
            self.say("Select a file first")
            return False
        notes = []
        for s in self.EDIT_SIDES:
            msg = self.guard(self.desk.save_side, self.nid, s,
                             get_text(self.editors[s]), self.loaded.get(s))
            if msg is None:
                return False
            self.loaded[s] = get_text(self.editors[s]).strip("\n")
            if msg != "no changes":
                notes.append(str(msg))
        self.refresh()
        self.say("; ".join(notes) or "no changes")
        return True

    def reload_file(self):
        if self.nid:
            self.load(self.nid)


class HumanWindow(EditorWindow):
    TITLE, SIZE, EDIT_SIDES = "Human concept", "980x640", ("human",)

    def build_editors(self):
        f, self.editors["human"] = text_box(self.right, height=6)
        f.pack(fill="both", expand=True)
        r = ttk.Frame(self.right)
        r.pack(fill="x")
        self.lang = Hint(r, "to", 14)
        bar(r, self.button(r, "Save", self.save),
            self.button(r, "Reload", self.reload_file),
            self.button(r, "Correct", self.correct, True),
            self.button(r, "Translate", self.translate, True), self.lang)

    def _apply(self, text):
        set_text(self.editors["human"], text)
        self.say("Applied to the editor. Save to keep it.")

    def correct(self):
        self.run_ai("Correct", "correct", get_text(self.editors["human"]),
                    lambda t: self.proposal.show_text(
                        "Proposed correction", t, self._apply))

    def translate(self):
        self.run_ai("Translate", "translate", get_text(self.editors["human"]),
                    lambda t: self.proposal.show_text(
                        "Proposed translation", t, self._apply),
                    lang=self.lang.get())


class CodeWindow(EditorWindow):
    TITLE, SIZE = "Machine language", "1120x780"
    EDIT_SIDES = ("instructions", "code")

    def build_editors(self):
        ttk.Label(self.right, text="Instructions for the model").pack(anchor="w")
        f, self.editors["instructions"] = text_box(self.right, height=3)
        f.pack(fill="both", expand=True)
        ttk.Label(self.right, text="Python code").pack(anchor="w")
        self.code = CodeText(self.right)
        self.code.pack(fill="both", expand=True)
        self.editors["code"] = self.code.text
        r = ttk.Frame(self.right)
        r.pack(fill="x")
        bar(r, self.button(r, "Save", self.save),
            self.button(r, "Reload", self.reload_file),
            self.button(r, "Check syntax", self.check),
            self.button(r, "Write code from instructions", self.write, True),
            self.button(r, "Fix code", self.fix, True),
            self.button(r, "Explain code", self.explain, True))

    def check(self):
        err = self.desk.check_code(get_text(self.editors["code"]))
        self.code.mark_error(err[0] if err else None)
        self.say("Syntax OK" if err is None
                 else "Syntax error on line %d: %s" % err)
        return err

    def _apply(self, text):
        set_text(self.editors["code"], text)
        self.say("Applied to the code editor. Check it, then Save.")

    def write(self):
        self.run_ai("Write code", "write_code",
                    get_text(self.editors["instructions"]),
                    lambda t: self.proposal.show_text(
                        "Proposed code (read it before applying)", t,
                        self._apply))

    def fix(self):
        err = self.desk.check_code(get_text(self.editors["code"]))
        extra = "Python reports: line %d: %s." % err if err else ""
        self.run_ai("Fix code", "fix_code", get_text(self.editors["code"]),
                    lambda t: self.proposal.show_text(
                        "Proposed fix (read it before applying)", t,
                        self._apply), extra=extra)

    def explain(self):
        self.run_ai("Explain", "explain_code", get_text(self.editors["code"]),
                    lambda t: self.proposal.show_text("Explanation", t))


# ------------------------------------------- window 4: search and schedule
class PlanWindow(DeskWindow):
    TITLE, SIZE = "Search and schedule", "1000x680"

    def build(self):
        top = ttk.Frame(self.body)
        top.pack(fill="x")
        self.query = Hint(top, "search trees", 30)
        self.query.bind("<Return>", lambda e: self.search())
        self.side_vars = {s: tk.BooleanVar(value=True) for s in SIDES}
        bar(top, self.query, self.button(top, "Search", self.search),
            *[ttk.Checkbutton(top, text=s, variable=v)
              for s, v in self.side_vars.items()])
        f, self.hits = table(self.body, [
            ("number", "No.", 90), ("tree", "Tree", 110),
            ("title", "Title", 160), ("side", "Side", 90),
            ("line", "Line", 50), ("text", "Text", 420)], height=8)
        f.pack(fill="both", expand=True, pady=6)
        self.hit_ids = {}
        self.hits.bind("<<TreeviewSelect>>", lambda e: self._hit_selected())
        ttk.Label(self.body, text="Schedule").pack(anchor="w")
        f, self.tasks = table(self.body, [
            ("title", "Task", 320), ("ref", "File or path", 240),
            ("due", "Due", 100), ("status", "Status", 110)], height=7)
        f.pack(fill="both", expand=True, pady=6)
        self.tasks.bind("<<TreeviewSelect>>", lambda e: self._task_selected())
        form = ttk.Frame(self.body)
        form.pack(fill="x")
        self.t_title = Hint(form, "task", 24)
        self.t_ref = Hint(form, "file or path", 16)
        self.t_due = Hint(form, "due YYYY-MM-DD", 11)
        self.t_status = ttk.Combobox(form, values=STATUSES, width=7,
                                     state="readonly")
        self.t_status.current(0)
        bar(form, self.t_title, self.t_ref, self.t_due, self.t_status)
        row = ttk.Frame(self.body)
        row.pack(fill="x")
        bar(row, self.button(row, "Add", self.add_task),
            self.button(row, "Update selected", self.update_task),
            self.button(row, "Mark done", self.mark_done),
            self.button(row, "Delete", self.delete_task),
            self.button(row, "Refresh", self.refresh))
        self.refresh()

    def chat_context(self):
        def lines(tree):
            return "\n".join(" | ".join(str(v) for v in tree.item(i, "values"))
                             for i in tree.get_children("")) or "(none)"
        return ("## Search results (number | tree | title | side | line | "
                "text)\n%s\n\n## Schedule (task | file or path | due | "
                "status)\n%s" % (lines(self.hits), lines(self.tasks)))

    def search(self):
        sides = [s for s, v in self.side_vars.items() if v.get()]
        rows = self.guard(self.desk.search, self.query.get(), sides)
        rows = [] if rows in (None, True) else rows
        self.hit_ids = {str(i): h["id"] for i, h in enumerate(rows)}
        fill(self.hits, [(str(i), (h["number"], h["tree"], h["title"],
                                   h["side"], h["line"], h["text"]))
                         for i, h in enumerate(rows)])
        self.say("%d match(es); select one to use it as a task's file"
                 % len(rows))

    def _hit_selected(self):
        key = selected(self.hits)
        if key in self.hit_ids:
            self.t_ref.set(self.hit_ids[key])

    def refresh(self):
        rows = self.guard(self.desk.tasks)
        rows = [] if rows in (None, True) else rows
        fill(self.tasks, [(t["id"], (
            t["title"], t["ref_label"], t["due"],
            t["status"] + (" (late)" if t["late"] else ""))) for t in rows])
        self.say("%d task(s), %d late" % (
            len(rows), sum(1 for t in rows if t["late"])))

    def _task_selected(self):
        tid = selected(self.tasks)
        for t in self.desk.tasks():
            if t["id"] == tid:
                self.t_title.set(t["title"])
                self.t_ref.set(t["ref"].partition(":")[2])
                self.t_due.set(t["due"])
                self.t_status.current(STATUSES.index(t["status"]))

    def _form(self):
        return dict(title=self.t_title.get(), ref=self.t_ref.get(),
                    due=self.t_due.get(), status=self.t_status.get())

    def _selected_task(self):
        tid = selected(self.tasks)
        if not tid:
            self.say("Select a task first")
        return tid

    def add_task(self):
        if self.guard(lambda: self.desk.add_task(**self._form())):
            self.refresh()

    def update_task(self):
        tid = self._selected_task()
        if tid and self.guard(lambda: self.desk.update_task(tid,
                                                            **self._form())):
            self.refresh()

    def mark_done(self):
        tid = self._selected_task()
        if tid and self.guard(lambda: self.desk.update_task(tid,
                                                            status="done")):
            self.refresh()

    def delete_task(self):
        tid = self._selected_task()
        if tid and self.guard(lambda: self.desk.delete_task(tid)):
            self.refresh()


WINDOWS = {"library": LibraryWindow, "human": HumanWindow,
           "code": CodeWindow, "plan": PlanWindow}


class App:
    def __init__(self, desk):
        self.project, self.windows = desk.project, []
        self.root = tk.Tk()
        self.root.withdraw()
        self.results = queue.Queue()
        self._poll()

    def _poll(self):
        try:
            while True:
                finish, res, err = self.results.get_nowait()
                finish(res, err)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def open(self, names):
        for name in names:
            self.windows.append(WINDOWS[name](self))
        return self.windows

    def closed(self, window):
        if window in self.windows:
            self.windows.remove(window)
        if not self.windows:
            self.root.quit()


def main(desk, names):
    app = App(desk)
    app.open(names)
    app.root.mainloop()
