#!/usr/bin/env python3
"""Tawjih desktop windows, for Linux users.

This part of the project is for Linux only. On Windows and other systems
use the terminal program, tawjih.py.

Four independent windows over a Tawjih project:
    library  deposit books and research; sort, search, translate, filter
    human    edit the human concept; correct, translate
    code     edit machine language: instructions and Python code
    plan     search all trees; schedule tasks on files and paths

AI help is off until "Use AI" is ticked in a window, and an answer is only
ever a proposal: nothing is saved until you apply it and press Save.

    python3 linux/tawjih_desk.py -p PROJECT            all four windows
    python3 linux/tawjih_desk.py -p PROJECT code plan  only these
    python3 linux/tawjih_desk.py --toolkit tk          force Tkinter
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from desk_core import THEMES, Desk, DeskError  # noqa: E402

NAMES = ["library", "human", "code", "plan"]
HELP = """No window toolkit was found. Install one of:
  GTK (recommended):  sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-gtksource-4
                      sudo dnf install python3-gobject gtk3 gtksourceview4
  Tkinter:            sudo apt install python3-tk    |    sudo dnf install python3-tkinter
and run this with the system Python (/usr/bin/python3)."""


def load_toolkit(choice):
    errors = []
    for name in (["gtk", "tk"] if choice == "auto" else [choice]):
        try:
            ui = __import__("desk_" + name)
            if name == "tk":
                print("Using Tkinter: Arabic text will be shown with unjoined "
                      "letters. Install GTK for Arabic content.",
                      file=sys.stderr)
            return ui, name
        except Exception as e:      # missing toolkit: try the next one
            errors.append("%s: %s" % (name, e))
    sys.exit(HELP + "\n\n" + "\n".join(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-p", "--project", default=".", help="project folder")
    p.add_argument("--toolkit", choices=["auto", "gtk", "tk"], default="auto")
    p.add_argument("--theme", choices=list(THEMES), default="dark",
                   help="dark or light colours, or the desktop's own "
                   "(default: dark)")
    p.add_argument("windows", nargs="*", metavar="WINDOW",
                   help="any of: %s (default: all)" % ", ".join(NAMES))
    a = p.parse_args()
    if not sys.platform.startswith("linux"):
        sys.exit("The desktop windows are for Linux users. On this system "
                 "use the terminal program: python tawjih.py --help")
    bad = [w for w in a.windows if w not in NAMES]
    if bad:
        p.error("unknown window: " + ", ".join(bad))
    try:
        desk = Desk(a.project)
    except DeskError as e:
        sys.exit("error: %s" % e)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        sys.exit("error: no graphical display found. Run this inside a "
                 "desktop session (over SSH, use: ssh -X).")
    ui, _ = load_toolkit(a.toolkit)
    ui.main(desk, a.windows or NAMES, a.theme)


if __name__ == "__main__":
    main()
