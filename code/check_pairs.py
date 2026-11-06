"""核对配对：同任务同种子的两局，初始场景是不是真的相同。比录像存的第一帧图像和第 0 步仿真状态。只读录像。"""
import json, os, sys
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
F = "/path/to/work/fwd"

def first(path):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    return m, z["sim_state"][0], [cv2.imdecode(z["q_" + c][0], cv2.IMREAD_COLOR).astype(np.float32) for c in ("left", "wrist", "right")]

def psnr(a, b):
    e = ((a - b) ** 2).mean()
    return 99.0 if e == 0 else float(10 * np.log10(255 ** 2 / e))

def one(a):
    d0, d1, n = a
    try:
        m0, s0, im0 = first(os.path.join(F, d0, "episodes", n))
        m1, s1, im1 = first(os.path.join(F, d1, "episodes", n))
    except Exception as e:
        return n, None, None, None, str(e), None, None
    sd = float(np.abs(s0 - s1).max()) if s0.shape == s1.shape else float("inf")
    return n, sd, min(psnr(x, y) for x, y in zip(im0, im1)), (m0["layout_id"], m0["style_id"]) == (m1["layout_id"], m1["style_id"]), None, bool(m0["success"]), bool(m1["success"])

SETS = [("rep_base_s30000", "rep_post_k3_s30000"), ("others_base_s40000", "others_post_k3_s40000"), ("rep_base_s30000", "rep_pblur06_s30000"), ("base_heldout_s7000", "test_post_k3_s7000"), ("base_seen_recheck_s7000", "test_post_k3_s7000"), ("base_p16_s20000", "pblur06_p16_s20000")]
out = {}
with ProcessPoolExecutor(16) as ex:
    for d0, d1 in SETS:
        ep0 = os.path.join(F, d0, "episodes")
        ep1 = os.path.join(F, d1, "episodes")
        if not (os.path.isdir(ep0) and os.path.isdir(ep1)):
            print(d0, d1, "没有录像"); continue
        names = sorted(set(os.listdir(ep0)) & set(os.listdir(ep1)))
        rows = list(ex.map(one, [(d0, d1, n) for n in names], chunksize=8))
        ok = [r for r in rows if r[1] is not None]
        same_state = sum(r[1] == 0 for r in ok)
        close = sum(r[1] < 1e-3 for r in ok)
        hi = sum(r[2] > 45 for r in ok)
        lo = [r for r in ok if r[2] < 30]
        print("%s vs %s：%d 对；初始状态完全相同 %d，差 < 1e-3 %d；首帧三路最低 PSNR > 45 dB %d，< 30 dB（场景不同）%d；layout/style 相同 %d；读取失败 %d" % (d0, d1, len(ok), same_state, close, hi, len(lo), sum(bool(r[3]) for r in ok), len(rows) - len(ok)), flush=True)
        out[d0 + "|" + d1] = [dict(name=r[0], max_state_diff=r[1], min_psnr=r[2], same_layout=bool(r[3]), s0=r[5], s1=r[6]) for r in ok]
        tasks = {}
        for r in lo:
            t = r[0].split("_base_")[0]
            tasks[t] = tasks.get(t, 0) + 1
        print("   场景不同的任务：", sorted(tasks.items(), key=lambda x: -x[1])[:30], flush=True)
json.dump(out, open("/path/to/work/blog/out/check_pairs.json", "w"))
print("CHECK PAIRS DONE")
