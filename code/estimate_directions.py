"""Estimate the directions removed by the post_k3 edit, and save them in the format of edit/edit.npz.

Input: an npz file of paired images, one row per simulator state, with PNG bytes in
raw_left, raw_wrist, raw_right (fresh renders) and mp4_left, mp4_wrist, mp4_right
(decoded training-video frames), all resized to 224 x 224 the way evaluation does.
pair_frames.py makes this file.

For every image pair: run SigLIP on both images, take the 256 tokens after post_layernorm,
subtract them, and add the differences to a second-moment matrix (the mean is not removed).
The directions are the top eigenvectors of that matrix.

load_policy() is left for you to fill in. It should return the PyTorch pi0.5 model and a function
that runs one image through the model's preprocessing and vision tower.

Usage: python estimate_directions.py pairs.npz out.npz --k 3
Then:  apply_edit(model, "out.npz")   # edit/apply_edit.py
"""
import argparse, json
import numpy as np
import cv2
import torch

CAMS = ("left", "wrist", "right")
PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"


def load_policy():
    """Return (model, embed). embed(img) takes a uint8 RGB image of 224 x 224 and runs the vision tower on it."""
    raise NotImplementedError("plug in your loader for the PyTorch pi0.5 here")


def dec(png_bytes):
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]


def second_moment(pairs_path, model, embed):
    vm = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model
    buf = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        buf["tokens"] = h[0].float().detach().cpu().numpy().astype(np.float64)

    handle = vm.post_layernorm.register_forward_hook(hook)
    z = np.load(pairs_path, allow_pickle=True)
    S, n = None, 0
    with torch.no_grad():
        for i in range(len(z["t"])):
            for c in CAMS:
                embed(dec(z["raw_" + c][i]))
                raw = buf["tokens"]
                embed(dec(z["mp4_" + c][i]))
                d = raw - buf["tokens"]
                S = d.T @ d if S is None else S + d.T @ d
                n += len(d)
    handle.remove()
    return S / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    model, embed = load_policy()
    S, n = second_moment(a.pairs, model, embed)
    w, v = np.linalg.eigh(S)
    V = v[:, ::-1][:, :a.k].astype(np.float32)
    np.savez(a.out, ops=json.dumps([{"name": PROJECTOR + ".weight", "kind": "input_abl", "alpha": 1.0}]), V_0=V)
    print("token pairs:", n, "share of the difference energy in the top directions:", float(w[::-1][:a.k].sum() / w.sum()))


if __name__ == "__main__":
    main()
