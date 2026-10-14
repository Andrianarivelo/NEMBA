"""Capture a Tk window on Windows without reading the screen (works when the
window is covered, the screen is locked or the monitor sleeps).

Uses the Win32 ``PrintWindow`` call with PW_RENDERFULLCONTENT, which asks the
window to paint itself into an off-screen bitmap. Falls back to
``PIL.ImageGrab`` on other platforms.
"""

import ctypes
import sys

from PIL import Image

PW_RENDERFULLCONTENT = 2


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


def capture_tk(root):
    """Return a PIL image of the client area of Tk window ``root``."""
    if sys.platform != "win32":
        from PIL import ImageGrab
        x, y = root.winfo_rootx(), root.winfo_rooty()
        return ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    from ctypes import wintypes
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = int(root.wm_frame(), 16)                    # top-level frame (with title bar)
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = user32.GetWindowDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    user32.PrintWindow(hwnd, mdc, PW_RENDERFULLCONTENT)
    bmi = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.SelectObject(mdc, old); gdi32.DeleteObject(bmp); gdi32.DeleteDC(mdc)
    user32.ReleaseDC(hwnd, hdc)
    img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
    # crop the title bar and borders: keep the client area of the Tk root
    left = root.winfo_rootx() - rect.left
    top = root.winfo_rooty() - rect.top
    return img.crop((left, top, left + root.winfo_width(), top + root.winfo_height()))


if __name__ == "__main__":                              # quick self-test
    import tkinter as tk
    r = tk.Tk(); r.geometry("320x120+40+40")
    tk.Label(r, text="PrintWindow capture test", bg="#2a78d6", fg="white",
             font=("Segoe UI", 14)).pack(fill="both", expand=True)
    for _ in range(20):
        r.update()
    im = capture_tk(r)
    import numpy as np
    print(im.size, "unique colours:", len(np.unique(np.asarray(im).reshape(-1, 3), axis=0)))
    r.destroy()
