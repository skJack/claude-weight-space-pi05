"""只动投影矩阵的少数几列（SigLIP 输出的少数几维）：看第 725 维是不是主角。对照是随机挑的、范数相近的列。"""
import json, os, struct
import numpy as np
JW = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
out = "/path/to/work/edits/d14"
os.makedirs(out, exist_ok=True)
def e(ids):
    V = np.zeros((1152, len(ids)), np.float32)
    for k, i in enumerate(ids):
        V[i, k] = 1.0
    return V
def save(name, ids, a=1.0):
    np.savez(f"{out}/{name}.npz", ops=json.dumps([{"name": JW + ".weight", "kind": "input_abl", "alpha": a}]), V_0=e(ids))
save("col725_a0.5", [725], 0.5)
save("col725_a1", [725], 1.0)
save("col725_a1.5", [725], 1.5)
save("col725_a2", [725], 2.0)
save("col725_961", [725, 961])
save("col725_961_242_788_655", [725, 961, 242, 788, 655])
save("col961", [961])
z = np.load("/path/to/work/read/fig/weights.npz", allow_pickle=True)
W0 = z["W0"].astype(np.float64)
cn = np.linalg.norm(W0, axis=0)
print("第 725 列范数 %.3f，全部列范数中位 %.3f，排名第 %d（从大到小）" % (cn[725], np.median(cn), int((cn > cn[725]).sum()) + 1))
rng = np.random.RandomState(0)
near = np.where(np.abs(cn - cn[725]) / cn[725] < 0.25)[0]
near = near[near != 725]
pick = (rng.choice(near, 2, replace=False) if len(near) >= 2 else rng.choice(1152, 2, replace=False)).tolist()
print("对照列（范数相近的随机列）", pick, np.round(cn[pick], 3))
for p in pick:
    save("col_ctrl%d" % p, [p])
print(sorted(os.listdir(out)))
