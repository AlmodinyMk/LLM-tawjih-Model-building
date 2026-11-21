"""GTK 3 front end: four independent windows over desk_core."""
import os
import threading

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

GtkSource = None
for _ver in ("4", "3.0"):
    try:
        gi.require_version("GtkSource", _ver)
        from gi.repository import GtkSource  # noqa: E402,F811
        break
    except (ValueError, ImportError):
        continue

from desk_core import (AI_BACKENDS, CHAT_CONTEXT_LIMIT, SIDES,  # noqa: E402
                       STATUSES, DeskError, ai_busy, palette)

WAITING = "Waiting for another AI request of this program to finish …"
SCHEME = None          # the code editor's colours, set by apply_theme
PAL = None             # the theme's colours, set by apply_theme

CSS = """
window, dialog, .background { background-color: @base; color: @text; }
label { color: @text; }
label.dim-label, .dim-label { color: @subtext; }
textview, textview text, treeview.view, entry, spinbutton, iconview {
  background-color: @mantle; color: @text; }
textview text selection, treeview.view:selected, entry selection,
  label selection {
  background-color: @accent; color: @on_accent; }
treeview.view:hover:not(:selected) { background-color: @surface0; }
treeview.view header button {
  background-image: none; background-color: @base; color: @subtext;
  border: none; border-bottom: 1px solid @surface0; box-shadow: none;
  font-weight: bold; padding: 4px 8px; }
scrolledwindow { border-radius: 8px; }
scrolledwindow, frame > border { border-color: @surface0; }
entry, spinbutton, combobox button.combo {
  background-image: none; border: 1px solid @surface1; border-radius: 8px;
  box-shadow: none; min-height: 30px; padding: 0 10px; }
entry:focus { border-color: @accent; box-shadow: 0 0 0 1px @accent; }
button {
  background-image: none; background-color: @surface0; color: @text;
  border: 1px solid @surface1; border-radius: 8px; box-shadow: none;
  text-shadow: none; padding: 5px 12px; min-height: 22px; }
button:hover { background-color: @surface1; border-color: @overlay0; }
button:active, button:checked { background-color: @surface2; }
button:disabled { background-color: @base; color: @overlay0;
  border-color: @surface0; }
button:disabled label { color: @overlay0; }
button.suggested-action { background-color: @accent; color: @on_accent;
  border-color: @accent; }
button.suggested-action label { color: @on_accent; }
check, radio { background-image: none; background-color: @mantle;
  border: 1px solid @overlay0; color: @on_accent; }
check:checked, radio:checked { background-color: @accent;
  border-color: @accent; }
scrollbar, scrollbar trough { background-color: transparent;
  border: none; }
scrollbar slider { background-color: @surface1; border-radius: 6px;
  min-width: 8px; min-height: 8px; border: none; }
scrollbar slider:hover { background-color: @overlay0; }
paned > separator, paned.wide > separator {
  background-image: none; background-color: @surface0; min-width: 4px;
  min-height: 4px; margin: 0 2px; }
paned > separator:hover { background-color: @accent; }
scrolledwindow undershoot, scrolledwindow overshoot {
  background-image: none; background-color: transparent; }
scrolledwindow > border, scrolledwindow.frame { border: 1px solid @surface0;
  border-radius: 8px; }
menu, .menu, popover, .popover { background-color: @mantle; color: @text;
  border: 1px solid @surface1; }
menuitem:hover { background-color: @accent; color: @on_accent; }
tooltip, tooltip * { background-color: @crust; color: @text; }
progressbar progress { background-color: @accent; }
treeview.view.files { -GtkTreeView-tree-line-width: 1;
  -GtkTreeView-tree-line-pattern: ""; border-color: @surface2; }
"""


def apply_theme(theme):
    """Dark or light colours for every window, or the desktop's own."""
    global SCHEME, PAL
    pal = palette(theme)
    if pal is None:
        return
    PAL = pal
    Gtk.Settings.get_default().set_property(
        "gtk-application-prefer-dark-theme", theme == "dark")
    css = "".join("@define-color %s %s;\n" % kv for kv in pal.items()) + CSS
    provider = Gtk.CssProvider()
    provider.load_from_data(css.encode("utf-8"))
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    if GtkSource is not None:
        SCHEME = _source_scheme(theme, pal)


def _source_scheme(theme, pal):
    """A GtkSourceView colour scheme from the same palette."""
    import tempfile
    sid = "tawjih-" + theme
    folder = os.path.join(tempfile.gettempdir(), "tawjih-styles-%d"
                          % os.getuid())
    os.makedirs(folder, exist_ok=True)
    styles = [("text", "text", "base"), ("selection", "on_accent", "accent"),
              ("current-line", None, "mantle"),
              ("line-numbers", "overlay0", "crust"),
              ("bracket-match", "text", "surface1"),
              ("def:comment", "overlay1", None), ("def:string", "green", None),
              ("def:number", "peach", None), ("def:keyword", "accent", None),
              ("def:statement", "accent", None), ("def:type", "yellow", None),
              ("def:function", "blue", None), ("def:builtin", "red", None),
              ("def:constant", "peach", None),
              ("def:special-char", "peach", None),
              ("def:preprocessor", "blue", None),
              ("def:error", "red", None)]
    rows = []
    for name, fg, bg in styles:
        attrs = ['name="%s"' % name]
        if fg:
            attrs.append('foreground="%s"' % pal[fg])
        if bg:
            attrs.append('background="%s"' % pal[bg])
        if name == "def:comment":
            attrs.append('italic="true"')
        rows.append("  <style %s/>" % " ".join(attrs))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<style-scheme id="%s" '
           'name="Tawjih %s" version="1.0">\n%s\n</style-scheme>\n'
           % (sid, theme, "\n".join(rows)))
    with open(os.path.join(folder, sid + ".xml"), "w",
              encoding="utf-8") as f:
        f.write(xml)
    mgr = GtkSource.StyleSchemeManager.get_default()
    if folder not in mgr.get_search_path():
        mgr.append_search_path(folder)
    mgr.force_rescan()
    return mgr.get_scheme(sid)


# ------------------------------------------------------------------ helpers
def scrolled(widget):
    s = Gtk.ScrolledWindow()
    s.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
    s.set_shadow_type(Gtk.ShadowType.IN)
    s.add(widget)
    s.set_hexpand(True)
    s.set_vexpand(True)
    return s


def text_view(editable=True, mono=False):
    v = Gtk.TextView()
    v.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
    v.set_editable(editable)
    v.set_monospace(mono)
    for side in ("left", "right", "top", "bottom"):
        getattr(v, "set_%s_margin" % side)(6)
    return v


def get_text(view):
    b = view.get_buffer()
    return b.get_text(b.get_start_iter(), b.get_end_iter(), True)


def set_text(view, text):
    """Replace the text. Not undoable: undo must never bring back the
    content of the file that was open before."""
    b = view.get_buffer()
    if hasattr(b, "begin_not_undoable_action"):
        b.begin_not_undoable_action()
        b.set_text(text)
        b.end_not_undoable_action()
    else:
        b.set_text(text)


def selection_or_all(view):
    b = view.get_buffer()
    if b.get_has_selection():
        s, e = b.get_selection_bounds()
        return b.get_text(s, e, True)
    return get_text(view)


def go_to_line(view, line):
    b = view.get_buffer()
    it = b.get_iter_at_line(max(line - 1, 0))
    b.place_cursor(it)
    view.scroll_to_iter(it, 0.1, True, 0.0, 0.3)


def add_columns(tree, titles):
    for i, title in enumerate(titles):
        col = Gtk.TreeViewColumn(title, Gtk.CellRendererText(), text=i)
        col.set_resizable(True)
        col.set_sort_column_id(i)
        tree.append_column(col)


def row(*widgets, spacing=6):
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=spacing)
    for w in widgets:
        box.pack_start(w, isinstance(w, Gtk.Entry), True, 0)
    return box


def button(label, handler):
    b = Gtk.Button(label=label)
    b.connect("clicked", lambda *_: handler())
    return b


def entry(placeholder, width=14):
    e = Gtk.Entry()
    e.set_placeholder_text(placeholder)
    e.set_width_chars(width)
    return e


# -------------------------------------------------------------- base window
class DeskWindow(Gtk.Window):
    """Common frame: AI bar on top, status line at the bottom."""
    TITLE, SIZE = "Tawjih", (900, 600)

    def __init__(self, app):
        super().__init__(title="Tawjih · " + self.TITLE)
        self.app, self.desk = app, app.desk
        self.set_default_size(*self.SIZE)
        self.ai_buttons = []
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.set_border_width(8)
        self.add(self.box)

        self.ai_on = Gtk.CheckButton(label="Use AI")
        self.ai_backend = Gtk.ComboBoxText()
        for b in AI_BACKENDS:
            self.ai_backend.append_text(b)
        self.ai_backend.set_active(0)
        self.ai_model = entry("model name", 18)
        self.ai_model.set_text(os.environ.get("TAWJIH_MODEL", ""))
        self.ai_on.connect("toggled", lambda *_: self._ai_state())
        self.chat = None
        self.box.pack_start(row(self.ai_on, Gtk.Label(label="provider"),
                                self.ai_backend, Gtk.Label(label="model"),
                                self.ai_model,
                                self.ai_button("Ask AI…", self.open_chat)),
                            False, False, 0)
        self.status = Gtk.Label(xalign=0)
        self.status.set_line_wrap(True)
        self.status.set_selectable(True)
        self.box.pack_end(self.status, False, False, 0)
        self.build()
        self._ai_state()
        self.alive = True
        self.connect("delete-event", lambda *_: self.closing())
        self.connect("destroy", lambda *_: self._destroyed())

    def build(self):
        raise NotImplementedError

    def closing(self):
        """Return True to keep the window open."""
        return False

    def _destroyed(self):
        self.alive = False
        if self.chat is not None:
            self.chat.destroy()
        self.app.closed(self)

    def open_chat(self):
        """The side conversation of this window (one per window)."""
        if self.chat is None:
            self.chat = ChatWindow(self)
            self.chat.show_all()
        self.chat.present()
        return self.chat

    def chat_context(self):
        """What this window currently shows, as text for the conversation."""
        return ""

    def say(self, text):
        self.status.set_text(text)

    def guard(self, fn, *args):
        """Run fn; show a DeskError in the status line instead of raising."""
        try:
            res = fn(*args)
        except DeskError as e:
            self.say("Error: %s" % e)
            return None
        return res if res is not None else True

    def confirm(self, text):
        d = Gtk.MessageDialog(transient_for=self, modal=True,
                              message_type=Gtk.MessageType.QUESTION,
                              buttons=Gtk.ButtonsType.YES_NO, text=text)
        ok = d.run() == Gtk.ResponseType.YES
        d.destroy()
        return ok

    # AI ------------------------------------------------------------------
    def ai_button(self, label, handler):
        b = button(label, handler)
        self.ai_buttons.append(b)
        return b

    def _ai_state(self, busy=False):
        on = self.ai_on.get_active()
        self.ai_backend.set_sensitive(on)
        self.ai_model.set_sensitive(on)
        for b in self.ai_buttons:
            b.set_sensitive(on and not busy)
        if self.chat is not None:
            self.chat.refresh_state()

    def ai_backend_name(self):
        return self.ai_backend.get_active_text()

    def ai_model_name(self):
        return self.ai_model.get_text().strip()

    def run_async(self, label, work, done):
        """Run work() off the main loop; call done(result) back on it."""
        self.say(WAITING if ai_busy() else label + " …")
        self._ai_state(busy=True)

        def finish(result, error):
            if not self.alive:
                return False
            self._ai_state()
            if error:
                self.say("Error: %s" % error)
            else:
                self.say(label + ": done")
                done(result)
            return False

        def target():
            try:
                res, err = work(), None
            except DeskError as e:
                res, err = None, str(e)
            except Exception as e:      # never lose an error in a thread
                res, err = None, "%s: %s" % (type(e).__name__, e)
            GLib.idle_add(finish, res, err)
        threading.Thread(target=target, daemon=True).start()

    def run_ai(self, label, action, text, done, **kw):
        if not self.ai_on.get_active():
            return
        backend = self.ai_backend_name()
        model = self.ai_model_name()
        self.run_async(label, lambda: self.desk.ai(action, text, backend,
                                                   model=model, **kw), done)


class ChatWindow(Gtk.Window):
    """A side conversation with the provider, next to one main window.
    It has no way to change the project: there is nothing to apply and
    nothing is saved, not even the conversation."""

    def __init__(self, owner):
        super().__init__(title="Tawjih · Ask AI · " + owner.TITLE)
        self.owner, self.history = owner, []
        self.busy, self.alive = False, True
        self.set_default_size(520, 600)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_border_width(8)
        self.add(box)
        note = Gtk.Label(xalign=0, label="A side conversation. Nothing here "
                         "is saved, and nothing here changes the project.")
        note.set_line_wrap(True)
        box.pack_start(note, False, False, 0)
        self.view = text_view(editable=False)
        self.view.get_buffer().create_tag("who", weight=700)
        box.pack_start(scrolled(self.view), True, True, 0)
        self.include = Gtk.CheckButton(
            label="Send what is open in the main window with my question")
        self.include.set_active(True)
        box.pack_start(self.include, False, False, 0)
        self.entry = text_view()
        self.entry.connect("key-press-event", self._key)
        holder = scrolled(self.entry)
        holder.set_vexpand(False)
        holder.set_size_request(-1, 80)
        box.pack_start(holder, False, False, 0)
        self.send_btn = button("Send  (Enter)", self.ask)
        box.pack_start(row(self.send_btn, button("Clear conversation",
                                                 self.reset)), False, False, 0)
        self.status = Gtk.Label(xalign=0)
        self.status.set_line_wrap(True)
        box.pack_start(self.status, False, False, 0)
        self.connect("destroy", lambda *_: self._closed())
        self.refresh_state()

    def _closed(self):
        self.alive = False
        if self.owner.chat is self:
            self.owner.chat = None

    def _key(self, widget, event):
        if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and not (
                event.state & Gdk.ModifierType.SHIFT_MASK):
            self.ask()
            return True
        return False

    def refresh_state(self):
        on = self.owner.ai_on.get_active()
        self.send_btn.set_sensitive(on and not self.busy)
        if not on:
            self.status.set_text("AI is off: tick \"Use AI\" in the main "
                                 "window to ask.")
        elif not self.busy:
            self.status.set_text("")

    def add_line(self, who, text):
        b = self.view.get_buffer()
        b.insert_with_tags_by_name(b.get_end_iter(), who + "\n", "who")
        b.insert(b.get_end_iter(), text + "\n\n")
        GLib.idle_add(lambda: self.view.scroll_to_iter(
            b.get_end_iter(), 0.0, False, 0, 0) and False)

    def reset(self):
        self.history = []
        set_text(self.view, "")

    def ask(self):
        question = get_text(self.entry).strip()
        if self.busy or not question or not self.owner.ai_on.get_active():
            return
        context = self.owner.chat_context() if self.include.get_active() \
            else ""
        backend, model = self.owner.ai_backend_name(), \
            self.owner.ai_model_name()
        history, desk, title = list(self.history), self.owner.desk, \
            self.owner.TITLE
        set_text(self.entry, "")
        self.add_line("You", question)
        self.busy = True
        self.refresh_state()
        self.status.set_text(WAITING if ai_busy() else "Asking …")

        def finish(result, error):
            if not self.alive:
                return False
            self.busy = False
            self.refresh_state()
            if error:
                self.status.set_text("Error: %s" % error)
                self.add_line("No answer", error)
                if not get_text(self.entry).strip():
                    set_text(self.entry, question)   # nothing to retype
                return False
            answer, cut = result
            self.history += [("User", question), ("Assistant", answer)]
            self.add_line("AI", answer)
            self.status.set_text(
                "Only the first %d characters of the window's content were "
                "sent." % CHAT_CONTEXT_LIMIT if cut else "")
            return False

        def target():
            try:
                res, err = desk.chat(question, backend, history, title,
                                     context, model), None
            except DeskError as e:
                res, err = None, str(e)
            except Exception as e:      # never lose an error in a thread
                res, err = None, "%s: %s" % (type(e).__name__, e)
            GLib.idle_add(finish, res, err)
        threading.Thread(target=target, daemon=True).start()


class Proposal(Gtk.Box):
    """Shows a model's answer. Nothing changes until the user applies it."""

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.label = Gtk.Label(xalign=0)
        self.view = text_view(editable=True)
        self.apply_btn = button("Apply", self._apply)
        self.pack_start(self.label, False, False, 0)
        self.pack_start(scrolled(self.view), True, True, 0)
        self.pack_start(row(self.apply_btn, button("Discard", self.clear)),
                        False, False, 0)
        self._cb = None
        self.clear()

    def show_text(self, title, text, apply_cb=None):
        self.label.set_markup("<b>%s</b>" % GLib.markup_escape_text(title))
        set_text(self.view, text)
        self._cb = apply_cb
        self.apply_btn.set_sensitive(apply_cb is not None)
        self.set_no_show_all(False)
        self.show_all()

    def _apply(self):
        if self._cb:
            self._cb(get_text(self.view))
            self.clear()

    def clear(self):
        self._cb = None
        self.set_no_show_all(True)
        self.hide()


class FileList(Gtk.TreeView):
    """The trees of the project: each file's code (its stable id), its
    number on lines joining it to its parent, its title and its sides;
    reports the selected file id."""

    SIDE_COLORS = {"H": "blue", "I": "green", "C": "peach", "R": "yellow"}

    def __init__(self, on_select):
        self.store = Gtk.TreeStore(str, str, str, str, str)
        super().__init__(model=self.store)
        self.get_style_context().add_class("files")
        self.set_enable_tree_lines(True)
        self.set_level_indentation(4)
        dim = PAL["overlay1"] if PAL else "gray"
        code = Gtk.CellRendererText(family="Monospace", foreground=dim,
                                    scale=0.9)
        num = Gtk.CellRendererText(family="Monospace", weight=600)
        title = Gtk.CellRendererText(xalign=1.0)
        sides = Gtk.CellRendererText(family="Monospace")
        cols = [Gtk.TreeViewColumn("Code", code, text=0),
                Gtk.TreeViewColumn("No.", num, text=1),
                Gtk.TreeViewColumn("Title", title, text=2),
                Gtk.TreeViewColumn("Sides", sides, markup=3)]
        cols[2].set_expand(True)
        for c in cols:
            c.set_resizable(True)
            self.append_column(c)
        self.set_expander_column(cols[1])
        self._filling, self._on_select = False, on_select
        self.get_selection().connect("changed", self._changed)

    def _sides(self, flags):
        out = []
        for c in flags:
            if c == "-":
                out.append('<span alpha="35%">·</span>')
            elif PAL:
                out.append('<span foreground="%s">%s</span>'
                           % (PAL[self.SIDE_COLORS[c]], c))
            else:
                out.append(c)
        return " ".join(out)

    def fill(self, rows, keep=None):
        self._filling = True
        self.store.clear()
        iters = {}
        for r in rows:
            iters[r["id"]] = self.store.append(
                iters.get(r["parent"]),
                [r["id"], r["number"], r["title"], self._sides(r["flags"]),
                 r["id"]])
        self.expand_all()
        if keep in iters:
            self.get_selection().select_iter(iters[keep])
        self._filling = False

    def select(self, nid):
        def visit(model, path, it):
            if model[it][4] == nid:
                self.get_selection().select_iter(it)
                return True
            return False
        self._filling = True
        self.store.foreach(visit)
        self._filling = False

    def _changed(self, selection):
        if self._filling:
            return
        model, it = selection.get_selected()
        if it is not None:
            self._on_select(model[it][4])


# ------------------------------------------------------- window 1: library
class LibraryWindow(DeskWindow):
    TITLE, SIZE = "Books and research", (1000, 680)

    def build(self):
        self.book = None
        self.query = entry("search in all deposited books", 30)
        self.query.connect("activate", lambda *_: self.search())
        self.box.pack_start(row(button("Deposit…", self.deposit),
                                button("Refresh", self.refresh), self.query,
                                button("Search", self.search)),
                            False, False, 0)

        self.books = Gtk.ListStore(str, str, str, int, str)
        self.books_view = Gtk.TreeView(model=self.books)
        add_columns(self.books_view, ("Id", "Name", "Date", "Characters",
                                      "State"))
        self.books_view.get_selection().connect("changed", self._book_changed)

        self.hits = Gtk.ListStore(str, str, int, str)
        self.hits_view = Gtk.TreeView(model=self.hits)
        add_columns(self.hits_view, ("Book", "Name", "Line", "Text"))
        self.hits_view.connect("row-activated", self._hit_activated)

        left = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        left.pack1(scrolled(self.books_view), True, False)
        left.pack2(scrolled(self.hits_view), True, False)
        left.set_position(260)

        self.text = text_view(editable=False)
        self.lang = entry("target language", 14)
        self.criteria = entry("extra filter criteria (optional)", 22)
        self.out = Proposal()
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right.pack_start(scrolled(self.text), True, True, 0)
        right.pack_start(row(
            self.ai_button("Translate selection to", self.translate),
            self.lang), False, False, 0)
        right.pack_start(row(
            self.ai_button("Filter into the trees", self.filter),
            self.criteria, button("Accept filter report", self.accept)),
            False, False, 0)
        right.pack_start(self.out, True, True, 0)

        paned = Gtk.Paned()
        paned.pack1(left, True, False)
        paned.pack2(right, True, False)
        paned.set_position(420)
        self.box.pack_start(paned, True, True, 0)
        self.refresh()

    def chat_context(self):
        if not self.book:
            return ""
        return "Deposited book %s:\n%s" % (self.book,
                                           selection_or_all(self.text))

    def refresh(self):
        rows = self.guard(self.desk.books)
        if rows is None or rows is True:
            rows = []
        self.books.clear()
        for b in rows:
            self.books.append([b["id"], b["name"], b["date"], b["chars"],
                               b["state"]])
        self.say("%d book(s) in the library" % len(rows))

    def deposit(self):
        d = Gtk.FileChooserDialog(title="Deposit a text book or paper",
                                  transient_for=self,
                                  action=Gtk.FileChooserAction.OPEN)
        d.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                      Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        path = d.get_filename() if d.run() == Gtk.ResponseType.OK else None
        d.destroy()
        if path:
            self.deposit_path(path)

    def deposit_path(self, path):
        msg = self.guard(self.desk.deposit, path)
        if msg:
            self.refresh()
            self.say(str(msg))

    def _book_changed(self, selection):
        model, it = selection.get_selected()
        if it is not None:
            self.load(model[it][0])

    def load(self, bid, line=None):
        text = self.guard(self.desk.book_text, bid)
        if text is None:
            return
        self.book = bid
        set_text(self.text, "" if text is True else text)
        if line:
            GLib.idle_add(lambda: go_to_line(self.text, line) and False)

    def search(self):
        rows = self.guard(self.desk.search_books, self.query.get_text())
        self.hits.clear()
        if rows in (None, True):
            rows = []
        for h in rows:
            self.hits.append([h["id"], h["name"], h["line"], h["text"]])
        self.say("%d line(s) found; double-click one to open it" % len(rows))

    def _hit_activated(self, view, path, column):
        r = self.hits[path]
        self.load(r[0], r[2])

    def translate(self):
        if not self.text.get_buffer().get_has_selection():
            return self.say("Select the passage to translate first")
        self.run_ai("Translate", "translate", selection_or_all(self.text),
                    lambda t: self.out.show_text("Translation", t),
                    lang=self.lang.get_text())

    def filter(self):
        if not self.book:
            return self.say("Select a book first")
        if not self.ai_on.get_active():
            return None
        backend, bid = self.ai_backend_name(), self.book
        ask, model = self.criteria.get_text(), self.ai_model_name()
        count = self.guard(self.desk.passage_count, bid)
        if count in (None, True) or not self.confirm(
                "Filtering sends this book to %s in %d passage(s), one call "
                "each. Continue?" % (backend, count)):
            return None

        def done(report):
            self.refresh()
            self.out.show_text("Filter report (nothing added yet; review, "
                               "then Accept)", report)
        self.run_async("Filter", lambda: self.desk.filter_book(
            bid, backend, ask, model), done)

    def accept(self):
        if not self.book:
            return self.say("Select a book first")
        msg = self.guard(self.desk.accept_book, self.book)
        if msg:
            self.refresh()
            self.out.show_text("Accepted into reading sides", str(msg))


# -------------------------------------------------- editors: windows 2 and 3
class EditorWindow(DeskWindow):
    """A file list on the left; subclasses add editors for their sides."""
    EDIT_SIDES = ()

    def build(self):
        self.nid, self.loaded = None, {}
        self.files = FileList(self.select)
        self.right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.heading = Gtk.Label(xalign=0)
        self.right.pack_start(self.heading, False, False, 0)
        self.editors = {}
        self.build_editors()
        self.proposal = Proposal()
        self.right.pack_start(self.proposal, True, True, 0)
        paned = Gtk.Paned()
        paned.pack1(scrolled(self.files), False, False)
        paned.pack2(self.right, True, False)
        paned.set_position(540)
        self.box.pack_start(paned, True, True, 0)
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
        """Before switching file or closing: offer to save changes."""
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
        n = self.desk.st.nodes[nid]
        self.heading.set_markup("<b>%s</b>  %s" % (
            GLib.markup_escape_text(self.desk.st.number(nid)),
            GLib.markup_escape_text(n.title)))
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
    TITLE, SIZE, EDIT_SIDES = "Human concept", (1220, 680), ("human",)

    def build_editors(self):
        self.editors["human"] = text_view()
        self.lang = entry("target language", 14)
        self.right.pack_start(scrolled(self.editors["human"]), True, True, 0)
        self.right.pack_start(row(
            button("Save", self.save), button("Reload", self.reload_file),
            self.ai_button("Correct", self.correct),
            self.ai_button("Translate to", self.translate), self.lang),
            False, False, 0)

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
                    lang=self.lang.get_text())


class CodeWindow(EditorWindow):
    TITLE, SIZE = "Machine language", (1300, 780)
    EDIT_SIDES = ("instructions", "code")

    def build_editors(self):
        self.editors["instructions"] = text_view()
        if GtkSource is not None:
            buf = GtkSource.Buffer()
            if SCHEME is not None:
                buf.set_style_scheme(SCHEME)
            buf.set_language(
                GtkSource.LanguageManager.get_default().get_language("python3")
                or GtkSource.LanguageManager.get_default().get_language("python"))
            buf.set_highlight_matching_brackets(True)
            code = GtkSource.View.new_with_buffer(buf)
            code.set_show_line_numbers(True)
            code.set_auto_indent(True)
            code.set_indent_width(4)
            code.set_tab_width(4)
            code.set_insert_spaces_instead_of_tabs(True)
            code.set_highlight_current_line(True)
            code.set_smart_backspace(True)
            code.set_monospace(True)
        else:
            code = text_view(mono=True)
            code.set_wrap_mode(Gtk.WrapMode.NONE)
        self.editors["code"] = code
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        top = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        top.pack_start(Gtk.Label(label="Instructions for the model", xalign=0),
                       False, False, 0)
        top.pack_start(scrolled(self.editors["instructions"]), True, True, 0)
        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        bottom.pack_start(Gtk.Label(label="Python code", xalign=0),
                          False, False, 0)
        bottom.pack_start(scrolled(code), True, True, 0)
        paned.pack1(top, True, False)
        paned.pack2(bottom, True, False)
        paned.set_position(200)
        self.right.pack_start(paned, True, True, 0)
        self.right.pack_start(row(
            button("Save", self.save), button("Reload", self.reload_file),
            button("Check syntax", self.check),
            self.ai_button("Write code from instructions", self.write),
            self.ai_button("Fix code", self.fix),
            self.ai_button("Explain code", self.explain)), False, False, 0)

    def check(self):
        err = self.desk.check_code(get_text(self.editors["code"]))
        if err is None:
            self.say("Syntax OK")
        else:
            go_to_line(self.editors["code"], err[0])
            self.say("Syntax error on line %d: %s" % err)
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
    TITLE, SIZE = "Search and schedule", (1000, 680)

    def build(self):
        self.query = entry("search in all trees", 30)
        self.query.connect("activate", lambda *_: self.search())
        self.side_checks = {}
        checks = []
        for s in SIDES:
            c = Gtk.CheckButton(label=s)
            c.set_active(True)
            self.side_checks[s] = c
            checks.append(c)
        self.hits = Gtk.ListStore(str, str, str, str, int, str, str)
        self.hits_view = Gtk.TreeView(model=self.hits)
        add_columns(self.hits_view, ("No.", "Tree", "Title", "Side", "Line",
                                     "Text"))
        self.hits_view.get_selection().connect("changed", self._hit_selected)
        top = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        top.pack_start(row(self.query, button("Search", self.search), *checks),
                       False, False, 0)
        top.pack_start(scrolled(self.hits_view), True, True, 0)

        self.tasks = Gtk.ListStore(str, str, str, str, str)
        self.tasks_view = Gtk.TreeView(model=self.tasks)
        add_columns(self.tasks_view, ("Task", "File or path", "Due", "Status"))
        self.tasks_view.get_selection().connect("changed", self._task_selected)
        self.t_title = entry("task", 26)
        self.t_ref = entry("file number/id or path name", 22)
        self.t_due = entry("due: YYYY-MM-DD", 14)
        self.t_status = Gtk.ComboBoxText()
        for s in STATUSES:
            self.t_status.append_text(s)
        self.t_status.set_active(0)
        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        bottom.pack_start(Gtk.Label(label="Schedule", xalign=0), False,
                          False, 0)
        bottom.pack_start(scrolled(self.tasks_view), True, True, 0)
        bottom.pack_start(row(self.t_title, self.t_ref, self.t_due,
                              self.t_status), False, False, 0)
        bottom.pack_start(row(button("Add", self.add_task),
                              button("Update selected", self.update_task),
                              button("Mark done", self.mark_done),
                              button("Delete", self.delete_task),
                              button("Refresh", self.refresh)),
                          False, False, 0)
        paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        paned.pack1(top, True, False)
        paned.pack2(bottom, True, False)
        paned.set_position(300)
        self.box.pack_start(paned, True, True, 0)
        self.refresh()

    def chat_context(self):
        hits = ["%s | %s | %s | %s | %s" % (r[0], r[2], r[3], r[4], r[5])
                for r in self.hits]
        tasks = ["%s | %s | due %s | %s" % (r[0], r[1], r[2] or "-", r[3])
                 for r in self.tasks]
        return ("## Search results (number | title | side | line | text)\n%s"
                "\n\n## Schedule (task | file or path | due | status)\n%s"
                % ("\n".join(hits) or "(none)", "\n".join(tasks) or "(none)"))

    def search(self):
        sides = [s for s, c in self.side_checks.items() if c.get_active()]
        rows = self.guard(self.desk.search, self.query.get_text(), sides)
        self.hits.clear()
        if rows in (None, True):
            rows = []
        for h in rows:
            self.hits.append([h["number"], h["tree"], h["title"], h["side"],
                              h["line"], h["text"], h["id"]])
        self.say("%d match(es); select one to use it as a task's file"
                 % len(rows))

    def _hit_selected(self, selection):
        model, it = selection.get_selected()
        if it is not None:
            self.t_ref.set_text(model[it][6])

    def refresh(self):
        rows = self.guard(self.desk.tasks)
        self.tasks.clear()
        if rows in (None, True):
            rows = []
        for t in rows:
            status = t["status"] + (" (late)" if t["late"] else "")
            self.tasks.append([t["title"], t["ref_label"], t["due"], status,
                               t["id"]])
        late = sum(1 for t in rows if t["late"])
        self.say("%d task(s), %d late" % (len(rows), late))

    def _selected_task(self):
        model, it = self.tasks_view.get_selection().get_selected()
        if it is None:
            self.say("Select a task first")
            return None
        return model[it][4]

    def _task_selected(self, selection):
        model, it = selection.get_selected()
        if it is None:
            return
        tid = model[it][4]
        for t in self.desk.tasks():
            if t["id"] == tid:
                self.t_title.set_text(t["title"])
                self.t_ref.set_text(t["ref"].partition(":")[2])
                self.t_due.set_text(t["due"])
                self.t_status.set_active(STATUSES.index(t["status"]))

    def _form(self):
        return dict(title=self.t_title.get_text(), ref=self.t_ref.get_text(),
                    due=self.t_due.get_text(),
                    status=self.t_status.get_active_text())

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
        self.desk_project, self.windows = desk.project, []
        self._first = desk

    @property
    def desk(self):
        """Each window gets its own view of the project."""
        from desk_core import Desk
        d, self._first = (self._first or Desk(self.desk_project)), None
        return d

    def open(self, names):
        for name in names:
            w = WINDOWS[name](self)
            self.windows.append(w)
            w.show_all()
        return self.windows

    def closed(self, window):
        if window in self.windows:
            self.windows.remove(window)
        if not self.windows:
            Gtk.main_quit()


def main(desk, names, theme="dark"):
    apply_theme(theme)
    App(desk).open(names)
    Gtk.main()
