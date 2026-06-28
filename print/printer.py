from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

try:
    import win32print  # type: ignore
except Exception:
    win32print = None


def list_printers() -> List[str]:
    if os.name != "nt" or win32print is None:
        return []
    flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
    printers = win32print.EnumPrinters(flags)
    return [item[2] for item in printers]


def print_image(
    image_path: str | Path,
    printer_name: Optional[str] = None,
    copies: int = 1,
) -> None:
    if os.name != "nt":
        raise OSError("Printing via os.startfile is only supported on Windows.")

    path = str(Path(image_path).resolve())
    copies = max(1, int(copies))

    if printer_name and win32print is not None:
        current_default = win32print.GetDefaultPrinter()
        try:
            win32print.SetDefaultPrinter(printer_name)
            for _ in range(copies):
                os.startfile(path, "print")
        finally:
            win32print.SetDefaultPrinter(current_default)
    else:
        for _ in range(copies):
            os.startfile(path, "print")
