# stage01
"""run_tests.py — точка входа для тестов digit_app.

Запускает pytest внутри .venv (если .venv есть), иначе — текущим
интерпретатором. Вызывается clipbridge автоматически после успешных
main-скриптов, а также вручную.
"""

import subprocess
import sys
from pathlib import Path


def _venv_python(root):
    if sys.platform == "win32":
        p = root / ".venv" / "Scripts" / "python.exe"
    else:
        p = root / ".venv" / "bin" / "python"
    return p if p.exists() else None


def main():
    root = Path(__file__).resolve().parent
    venv_py = _venv_python(root)
    interpreter = str(venv_py) if venv_py else sys.executable
    cmd = [interpreter, "-m", "pytest", "tests", "-v", "--tb=short"]
    return subprocess.run(cmd, cwd=str(root)).returncode


if __name__ == "__main__":
    sys.exit(main())
