# Tawjih desktop windows, for Linux users

**This version is for Linux users only.** It needs a Linux desktop session
and has been run only on Linux. On Windows and other systems use the
terminal program `tawjih.py`, which works on both.

Four independent windows over a Tawjih project. Each one can be opened on
its own, and closing one leaves the others running.

| Window | Opens with | What it does |
|---|---|---|
| Books and research | `library` | Deposit text books and papers; sort the list by any column; search inside all books; translate a selection; filter a book into the trees and accept the report |
| Human concept | `human` | Edit the human side of a file; correct it; translate it |
| Machine language | `code` | Edit the instructions and the Python code of a file, with line numbers, colours, auto-indent, 4-space tabs and a syntax check; write code from the instructions, fix it, explain it |
| Search and schedule | `plan` | Search every side of every tree; keep a list of tasks, each linked to a file or a path, with a due date and a status |

```
python3 linux/tawjih_desk.py -p PROJECT              # all four windows
python3 linux/tawjih_desk.py -p PROJECT code plan    # only these two
python3 linux/tawjih_desk.py -p PROJECT --toolkit tk # force Tkinter
python3 linux/tawjih_desk.py -p PROJECT --theme light   # light colours
```

## Themes

`--theme dark` (the default) and `--theme light` colour every window, both
toolkits and the code editor alike, with the free
[Catppuccin](https://github.com/catppuccin/palette) palettes (Mocha and
Latte, MIT licence). Nothing is installed for them: GTK gets a style sheet
and a GtkSourceView colour scheme, Tkinter a ttk style. `--theme system`
keeps the desktop's own look, as before.

The list of files in the human and machine-language windows has four
columns: **Code** (the file's stable id), **No.** (its number, on lines
joining it to its parent: ▾ marks a file with files under it), **Title**
(right-aligned for Arabic), and **Sides** (H human, I instructions, C code,
R reading; a dot for an empty side, each letter in its own colour in GTK).
In Tkinter the lines are drawn with ├─ └─ │, and the tree cannot be folded.

## Toolkits

Two front ends give the same four windows. GTK is used when it is available,
Tkinter otherwise.

| | GTK 3 | Tkinter |
|---|---|---|
| Install (Debian, Ubuntu) | `sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-gtksource-4` | `sudo apt install python3-tk` |
| Install (Fedora) | `sudo dnf install python3-gobject gtk3 gtksourceview4` | `sudo dnf install python3-tkinter` |
| Arabic text | Shown correctly | **Letters are not joined** (a Tk limitation on Linux) |
| Code editor | GtkSourceView | Built in, simpler |

Use the system Python (`/usr/bin/python3`): the GTK bindings are installed
for it and are usually not visible from a virtual environment.

## AI help

AI is off until you tick **Use AI** in a window and choose the provider and
the model. The providers are the ones `tawjih.py` uses (`ollama`, `openai`,
`anthropic`), configured by the same environment variables.

An answer is only ever a proposal. It appears in a separate pane, and
nothing changes until you press **Apply**, and nothing is written to the
file until you press **Save**. Filtering a book writes a report only; the
trees change when you press **Accept filter report**.

### Ask AI: the side conversation

Every window has an **Ask AI…** button, active once Use AI is ticked. It
opens a second window beside that one, where you write to the provider and
read its answers. One side window per main window.

- It cannot change the project. It has no Apply and no Save, and the
  conversation itself is not stored: closing the window discards it.
- With "Send what is open in the main window" ticked (the default), your
  question goes with that window's content: the selected passage or open
  book, the file's sides as they stand in the editors, or the search results
  and schedule. Untick it to send the question alone.
- At most 30,000 characters of content and the last 12 messages are sent.
- Enter sends, Shift+Enter starts a new line.

Text you send goes to the provider you chose. Use `ollama` for private
material.

## Notes

- Tasks are kept in `tasks.json` in the project. Scheduling is a list with
  due dates and "late" marks; nothing runs automatically.
- Windows re-read a file before saving one side, so saving the human side in
  one window and the code in another does not lose either.
- If the side you are editing was changed elsewhere after you opened it,
  Save refuses and tells you to Reload; it never overwrites the other change.
- In the library, Translate needs a selection, and Filter first tells you how
  many passages (one provider call each) it will send and asks to continue.
- Provider calls run one at a time across the windows of one process.
- The interface is in English. Content can be in any language.
- Every save keeps the previous version in `history/`.

## What is verified

The logic (`desk_core.py`) is covered by `tests/test_desk.py`. Both front
ends were driven by a script on a virtual display (Xvfb) with a stand-in for
the model: every button's action was called and its effect on the project
checked. They have not been used by a person on a real desktop, and no real
AI provider was called.
