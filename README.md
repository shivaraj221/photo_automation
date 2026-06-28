# Passport Photo Automation Tool

Production-focused desktop tool for passport photo processing and print-ready 4x6 output.

## Features

- Face detection with MediaPipe Face Mesh
- Face detection with FaceNet MTCNN (facenet-pytorch)
- Eye-level alignment (horizontal eye line correction)
- Dynamic passport crop based on real dimensions (mm -> px at 300 DPI)
- Clean background replacement pipeline:
- Soft mask refinement + feathered edges
- Optional U2Net matting (if `rembg` is installed)
- Foreground brightness/color balancing against white background
- Subtle retouching:
- Bilateral smoothing
- CLAHE + mild gamma lighting correction
- 4x6 sheet generator at 1200x1800 px (300 DPI)
- Copy presets: 4, 6, 8
- Windows print trigger with printer selection support
- Tkinter desktop UI with preview

## Project Structure

```text
project/
│
├── main.py
├── config.py
├── pipeline.py
├── face/
│   ├── detect.py
│   ├── align.py
│   └── crop.py
│
├── enhance/
│   ├── smooth.py
│   └── lighting.py
│
├── layout/
│   └── grid.py
│
├── print/
│   └── printer.py
│
└── ui/
    └── app.py
```

## Install

```bash
pip install -r requirements.txt
```

Optional for improved hair-edge matting:

```bash
pip install onnxruntime
pip install --no-deps rembg
```

## Run

To launch the Passport Photo Studio:

```bash
run_studio.bat
```

Or manually:
```bash
python main.py
```



## Print Accuracy Rules

- Set paper size to `4 x 6 in`.
- Disable scaling (`Actual Size` / `100%` / `No Fit`).
- Keep output at 300 DPI.
