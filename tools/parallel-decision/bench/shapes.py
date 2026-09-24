#!/usr/bin/env python3
# Synthetic labelled images for vision decisions: coloured shapes on a plain background.
# Writes OUT/NNNN.png and OUT/labels.jsonl ({"image": ..., "labels": {...}}), plus OUT/schema.json.
# --unknown-rate R draws a share of images without shapes; shape and color then accept "unknown" and count 0.
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (235, 200, 30)}
SHAPES = ["circle", "square", "triangle"]

SCHEMA = {
    "shape":      {"type": "enum", "choices": SHAPES, "description": "Which shape is drawn?"},
    "color":      {"type": "enum", "choices": list(COLORS), "description": "What colour are the shapes?"},
    "count":      {"type": "integer", "minimum": 1, "maximum": 4, "description": "How many shapes are there?"},
    "dark_background": {"type": "boolean", "description": "Is the background dark?"},
}


def draw(rng: random.Random, size: int, unknown_rate: float = 0.0):
    shape, color, count = rng.choice(SHAPES), rng.choice(list(COLORS)), rng.randint(1, 4)
    dark = rng.random() < 0.5
    img = Image.new("RGB", (size, size), (25, 25, 30) if dark else (240, 240, 235))
    if unknown_rate > 0 and rng.random() < unknown_rate:
        return img, {"shape": "unknown", "color": "unknown", "count": 0, "dark_background": dark}
    d = ImageDraw.Draw(img)
    cell = size // 2
    slots = rng.sample(range(4), count)
    for s in slots:
        cx, cy = (s % 2) * cell + cell // 2, (s // 2) * cell + cell // 2
        r = rng.randint(cell // 4, cell // 3)
        box = (cx - r, cy - r, cx + r, cy + r)
        if shape == "circle":
            d.ellipse(box, fill=COLORS[color])
        elif shape == "square":
            d.rectangle(box, fill=COLORS[color])
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[color])
    return img, {"shape": shape, "color": color, "count": count, "dark_background": dark}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unknown-rate", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    with open(os.path.join(args.out, "labels.jsonl"), "w") as f:
        for i in range(args.n):
            img, labels = draw(rng, args.size, args.unknown_rate)
            name = f"{i:04d}.png"
            img.save(os.path.join(args.out, name))
            f.write(json.dumps({"image": name, "labels": labels}) + "\n")
    schema = json.loads(json.dumps(SCHEMA))
    if args.unknown_rate > 0:
        schema["shape"]["choices"].append("unknown")
        schema["color"]["choices"].append("unknown")
        schema["count"]["minimum"] = 0
        schema["shape"]["description"] = "Which shape is drawn? unknown if there is no shape."
        schema["color"]["description"] = "What colour are the shapes? unknown if there is no shape."
    with open(os.path.join(args.out, "schema.json"), "w") as f:
        json.dump(schema, f, indent=1)


if __name__ == "__main__":
    main()
