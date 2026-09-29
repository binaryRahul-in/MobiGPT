#!/usr/bin/env python3
"""Renders the MobiGPT app icon for Android and iOS (run from repo root)."""
import json
import os

from PIL import Image, ImageDraw

S = 1024
V = (0x6D, 0x5B, 0xFF)
T = (0x00, 0xC9, 0xA7)


def gradient(size):
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(int(V[i] + (T[i] - V[i]) * t) for i in range(3))
    return img


def mark(size, rounded=True, circle=False, pad=0.0):
    base = gradient(size).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    inset = int(size * pad)
    if circle:
        md.ellipse([inset, inset, size - inset, size - inset], fill=255)
    elif rounded:
        md.rounded_rectangle([inset, inset, size - inset, size - inset], radius=int(size * 0.26), fill=255)
    else:
        md.rectangle([0, 0, size, size], fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(base, (0, 0), mask)
    d = ImageDraw.Draw(out)
    k = size / 100.0
    w = int(9 * k)
    pts = [(24, 68), (24, 38), (50, 60), (76, 38), (76, 68)]
    pts = [(x * k, y * k) for x, y in pts]
    d.line(pts, fill="white", width=w, joint="curve")
    for x, y in (pts[0], pts[-1]):
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill="white")
    r = 6 * k
    d.ellipse([78 * k - r, 22 * k - r, 78 * k + r, 22 * k + r], fill=(255, 255, 255, 230))
    return out


def main():
    big = mark(S, rounded=False)  # iOS applies its own mask
    ios_dir = "ios/MobiGPT/Images.xcassets/AppIcon.appiconset"
    big.convert("RGB").save(os.path.join(ios_dir, "AppIcon-1024.png"))
    json.dump(
        {
            "images": [{"filename": "AppIcon-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
            "info": {"author": "xcode", "version": 1},
        },
        open(os.path.join(ios_dir, "Contents.json"), "w"),
        indent=2,
    )
    sizes = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
    square = mark(S, rounded=True, pad=0.04)
    round_ = mark(S, circle=True, pad=0.02)
    for d, px in sizes.items():
        folder = f"android/app/src/main/res/mipmap-{d}"
        square.resize((px, px), Image.LANCZOS).save(f"{folder}/ic_launcher.png")
        round_.resize((px, px), Image.LANCZOS).save(f"{folder}/ic_launcher_round.png")
    mark(512, rounded=True).save("assets/brand/mobigpt-icon-512.png")
    print("icons written")


if __name__ == "__main__":
    main()
