# stage07
"""digit_app — точка входа."""

import sys
import tkinter as tk
from tkinter import messagebox

from digit_app.ui import App


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as e:
        # Если запускать без графической среды (SSH, CI) — сообщаем.
        print(f"Не удалось создать окно Tk: {e}", file=sys.stderr)
        return 1
    try:
        App(root)
    except Exception as e:
        messagebox.showerror("Запуск", f"Не удалось инициализировать UI: {e}")
        raise
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
