from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk

from config import FRAMING_PRESETS, OUTPUT_DIR, PASSPORT_CONFIG, SUPPORTED_COPIES
from pipeline import process_passport_photo
from print.printer import list_printers, print_image

# Colors
_BG = "#f5f5f7"
_FG = "#1d1d1f"
_ACCENT = "#0071e3"

class PassportPhotoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Passport Photo AI Studio")
        self.geometry("1000x750")
        self.configure(bg=_BG)

        self.input_path = None
        self.sheet_path = None
        self.passport_path = None
        self._img_refs = {} 

        # Variables
        self.copy_var = tk.StringVar(value=str(SUPPORTED_COPIES[0]))
        self.printer_var = tk.StringVar(value="System default printer")
        self.status_var = tk.StringVar(value="Ready. Open a photo to start.")
        self.file_label_var = tk.StringVar(value="No file selected")
        self.force_ai_var = tk.BooleanVar(value=False) # Speed vs Quality toggle

        self._setup_ui()
        self._load_printers()
        self._q = queue.Queue()
        self.after(100, self._process_queue)

    def _setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background=_BG)
        style.configure("TLabel", background=_BG, foreground=_FG, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=_BG, font=("Segoe UI", 9))

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        ttk.Label(main_frame, text="Passport Photo AI Studio", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 20))

        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(content, width=280)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        controls.pack_propagate(False)

        previews = ttk.Frame(content)
        previews.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Controls ---
        ttk.Button(controls, text="1. Open Photo", command=self.select_file).pack(fill=tk.X, pady=5)
        ttk.Label(controls, textvariable=self.file_label_var, wraplength=250, font=("Segoe UI", 9)).pack(fill=tk.X, pady=(0, 15))

        ttk.Label(controls, text="Copies on Sheet:").pack(anchor=tk.W)
        ttk.Combobox(controls, textvariable=self.copy_var, values=[str(c) for c in SUPPORTED_COPIES], state="readonly").pack(fill=tk.X, pady=5)

        ttk.Label(controls, text="Printer:").pack(anchor=tk.W, pady=(10, 0))
        self.printer_cb = ttk.Combobox(controls, textvariable=self.printer_var, state="readonly")
        self.printer_cb.pack(fill=tk.X, pady=5)

        # Quality Toggle
        ttk.Label(controls, text="Enhancement:").pack(anchor=tk.W, pady=(15, 0))
        ttk.Checkbutton(controls, text="High Quality AI Boost", variable=self.force_ai_var).pack(anchor=tk.W, pady=5)
        ttk.Label(controls, text="(Turn OFF for faster processing)", font=("Segoe UI", 8), foreground="#888").pack(anchor=tk.W)

        self.btn_process = ttk.Button(controls, text="2. Process Photo", style="Action.TButton", command=self.process_photo)
        self.btn_process.pack(fill=tk.X, pady=(25, 5))

        self.btn_print = ttk.Button(controls, text="3. Print Sheet", state=tk.DISABLED, command=self.print_sheet)
        self.btn_print.pack(fill=tk.X, pady=5)

        ttk.Label(controls, textvariable=self.status_var, wraplength=250, foreground="#666").pack(fill=tk.X, pady=20)

        # --- Previews ---
        preview_area = ttk.Frame(previews)
        preview_area.pack(fill=tk.BOTH, expand=True)

        pass_frame = ttk.LabelFrame(preview_area, text="Passport Crop (Review)", padding=10)
        pass_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.lbl_pass = ttk.Label(pass_frame, text="Cropped photo will appear here", anchor=tk.CENTER, justify=tk.CENTER, background="#fff")
        self.lbl_pass.pack(fill=tk.BOTH, expand=True)

        sheet_frame = ttk.LabelFrame(preview_area, text="4x6 Print Sheet (Review)", padding=10)
        sheet_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lbl_sheet = ttk.Label(sheet_frame, text="Print sheet will appear here", anchor=tk.CENTER, justify=tk.CENTER, background="#fff")
        self.lbl_sheet.pack(fill=tk.BOTH, expand=True)

    def _load_printers(self):
        printers = list_printers()
        self.printer_cb["values"] = ["System default printer"] + printers
        self.printer_cb.current(0)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")])
        if path:
            self.input_path = path
            self.file_label_var.set(Path(path).name)
            self.status_var.set("Photo selected. Click 'Process Photo'.")

    def process_photo(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Select a photo first.")
            return
        
        self.btn_process.configure(state=tk.DISABLED)
        self.btn_print.configure(state=tk.DISABLED)
        
        status_text = "⏳ AI Processing... Please wait." if self.force_ai_var.get() else "⏳ Fast Processing..."
        self.status_var.set(status_text)
        
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            results = process_passport_photo(
                input_path=self.input_path,
                copies=int(self.copy_var.get()),
                force_ai=self.force_ai_var.get()
            )
            self._q.put(("done", results))
        except Exception as e:
            traceback.print_exc()
            self._q.put(("error", str(e)))

    def _process_queue(self):
        try:
            while True:
                kind, data = self._q.get_nowait()
                if kind == "done":
                    self.passport_path = data["passport_path"]
                    self.sheet_path = data["sheet_path"]
                    self._update_images()
                    self.btn_print.configure(state=tk.NORMAL)
                    self.status_var.set("✅ Done! Click 'Print Sheet' to finish.")
                elif kind == "error":
                    messagebox.showerror("Error", f"Processing failed: {data}")
                    self.status_var.set("❌ Error occurred.")
                self.btn_process.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.after(100, self._process_queue)

    def _update_images(self):
        if self.passport_path:
            try:
                img = Image.open(self.passport_path)
                img.thumbnail((320, 420), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_refs['pass'] = photo 
                self.lbl_pass.configure(image=photo, text="")
            except Exception as e:
                print(f"Preview error: {e}")

        if self.sheet_path:
            try:
                img = Image.open(self.sheet_path)
                img.thumbnail((450, 650), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_refs['sheet'] = photo 
                self.lbl_sheet.configure(image=photo, text="")
            except Exception as e:
                print(f"Preview error: {e}")

    def print_sheet(self):
        if not self.sheet_path: return
        printer = self.printer_var.get()
        if printer == "System default printer": printer = None
        try:
            print_image(self.sheet_path, printer_name=printer)
            messagebox.showinfo("Success", "Print job sent. Ensure paper is 4x6.")
        except Exception as e:
            messagebox.showerror("Print Error", str(e))

def launch_app():
    app = PassportPhotoApp()
    app.mainloop()

if __name__ == "__main__":
    launch_app()
