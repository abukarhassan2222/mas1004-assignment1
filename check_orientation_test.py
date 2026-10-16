"""Reproduce the EXIF-rotation discrepancy between check.py and the demo page.

The model was trained on upright photos (data/clean has no sideways images),
but five of the trafalgar_law_figure new images are stored sideways (EXIF
orientation 6). Python's prepare_image and check.py's prepare do NOT apply the
EXIF orientation, so they feed the model the sideways pixels. The browser
applies the EXIF orientation automatically, so the demo page feeds the model
the upright pixels. That is why check.py reports 3/10 for trafalgar_law_figure
while the demo page reports more.

This script runs the exported model (docs/model.onnx, the same file the browser
runs) on the trafalgar_law_figure new images three ways:

    as stored   what check.py feeds          -> 3/10
    upright     what the browser feeds       -> 7/10
    changed     the copies in data/changed/  -> 2/10

Run it with:

    uv run python check_orientation_test.py

It needs the data/ folder, which is not in git. Copy data/ from the student's
machine (or run it there) to reproduce the numbers above.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "docs" / "model.json"
MODEL_ONNX = ROOT / "docs" / "model.onnx"
NEW_IMAGES = ROOT / "data" / "new_images" / "trafalgar_law_figure"
CHANGED = ROOT / "data" / "changed" / "trafalgar_law_figure_no_head"
CLASS = "trafalgar_law_figure"
SUFFIXES = {".jpg", ".jpeg", ".png"}


def prepare(image, settings):
    """The six steps of check.py's prepare (src/check.py lines 231-249)."""
    image = image.convert("RGB")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    resize = settings["resize"]
    new_long = int(resize * long / short)
    new_size = (resize, new_long) if width <= height else (new_long, resize)
    image = image.resize(new_size, Image.BILINEAR)
    crop = settings["crop"]
    left = int(round((image.width - crop) / 2.0))
    top = int(round((image.height - crop) / 2.0))
    image = image.crop((left, top, left + crop, top + crop))
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = (values - np.array(settings["mean"], dtype=np.float32)) / np.array(
        settings["std"], dtype=np.float32
    )
    return values.transpose(2, 0, 1).astype(np.float32)


def run(folder, transpose, session, labels, settings):
    """Classify every image in folder; return (right, total)."""
    right = total = 0
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in SUFFIXES:
            continue
        image = Image.open(p)
        if transpose:
            image = ImageOps.exif_transpose(image)
        x = prepare(image, settings)
        logits = session.run(None, {"image": x[None]})[0][0]
        guess = int(np.argmax(logits))
        correct = guess == labels.index(CLASS)
        right += correct
        total += 1
        print(f"  {p.name:16s} -> {labels[guess]:24s} {'OK' if correct else 'WRONG'}")
    return right, total


def main():
    if not MODEL_JSON.exists() or not MODEL_ONNX.exists():
        raise SystemExit("docs/model.json and docs/model.onnx are missing. "
                         "Run `uv run python src/export_web.py --tag clean` first.")
    if not NEW_IMAGES.exists():
        raise SystemExit(f"{NEW_IMAGES} is missing. The data/ folder is not in "
                         "git; copy it from the student's machine.")

    model = json.loads(MODEL_JSON.read_text())
    session = onnxruntime.InferenceSession(
        str(MODEL_ONNX), providers=["CPUExecutionProvider"]
    )
    labels = model["labels"]
    settings = model["input"]

    for title, folder, transpose in [
        ("as stored (what check.py feeds)", NEW_IMAGES, False),
        ("upright (what the browser feeds)", NEW_IMAGES, True),
    ]:
        print(f"--- {title} ---")
        right, total = run(folder, transpose, session, labels, settings)
        print(f"{title}: {right}/{total}\n")

    if CHANGED.exists():
        print("--- changed copies (upright + grey band) ---")
        right, total = run(CHANGED, False, session, labels, settings)
        print(f"changed copies: {right}/{total}\n")


if __name__ == "__main__":
    main()
