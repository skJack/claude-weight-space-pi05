"""Copy rendered clips and image pairs into assets/ (film strip frames, amplified differences, share card).
Usage: python3 prepare_assets.py <dir with rendered clips> <dir with image pairs> <repo root>"""
import glob, json, os, shutil, sys
import numpy as np
from PIL import Image

VID, PAIRS, ROOT = sys.argv[1:4]
A = os.path.join(ROOT, "assets")
for sub in ("video", "img/pairs", "img/film"):
    os.makedirs(os.path.join(A, sub), exist_ok=True)
for f in glob.glob(os.path.join(VID, "*")):
    if f.endswith((".mp4", "_poster.jpg")) or (f.endswith(".json")):
        shutil.copy(f, os.path.join(A, "video", os.path.basename(f)))
NAMES = {"potato": "PickPlaceCounterToOven_166_wrist", "onion": "PickPlaceCounterToStove_179_wrist", "hotdog": "PickPlaceToasterOvenToCounter_235_wrist"}
for k, src in NAMES.items():
    raw = Image.open(os.path.join(PAIRS, src + "_raw.png")).convert("RGB")
    mp4 = Image.open(os.path.join(PAIRS, src + "_mp4.png")).convert("RGB")
    raw.save(os.path.join(A, "img/pairs", k + "_raw.png"), optimize=True)
    mp4.save(os.path.join(A, "img/pairs", k + "_mp4.png"), optimize=True)
    d = np.clip(np.abs(np.asarray(raw, float) - np.asarray(mp4, float)) * 8, 0, 255).astype(np.uint8)
    Image.fromarray(d).save(os.path.join(A, "img/pairs", k + "_diff.png"), optimize=True)
z = np.load(os.path.join(VID, "coffee_film.npz"))
for k in z.files:
    Image.fromarray(z[k]).resize((320, 320), Image.LANCZOS).save(os.path.join(A, "img/film", k + ".jpg"), quality=84)
# share card: the same moment in both episodes, once the edited run has finished
card = np.hstack([z["b8"], np.full((512, 8, 3), 255, np.uint8), z["e8"]])
Image.fromarray(card).save(os.path.join(A, "img/card.jpg"), quality=84)
print(sorted(os.listdir(os.path.join(A, "video")))[:6], len(os.listdir(os.path.join(A, "img/film"))))
