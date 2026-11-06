"""Write the public CSV files in results/ from the raw experiment logs. Run once, from the project that holds experiment/.
Usage: python3 make_results.py <path to experiment/> <path to blog/>"""
import csv, json, os, re, sys
from math import comb
import numpy as np

EXP, OUT = sys.argv[1], os.path.join(sys.argv[2], "results")
F = os.path.join(EXP, "wsp_results", "fwd")


def load(n):
    o = {}
    for l in open(os.path.join(F, n, "results.jsonl")):
        r = json.loads(l)
        if "error" in r or "success" not in r:
            continue
        o[(r["task"], r["episode"])] = r
    return o


def mc(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


CK = json.load(open(os.path.join(EXP, "wsp_results", "check_pairs.json")))
exact = {}
for key, rows in CK.items():
    exact[key] = {r["name"].replace(".npz", ""): r["max_state_diff"] == 0 for r in rows}


def is_exact(key, task, ep):
    return exact[key].get("%s_base_ep%d" % (task, ep))


# fresh seeds
rb, rv, rk = load("rep_base_s30000"), load("rep_pblur06_s30000"), load("rep_post_k3_s30000")
ob, ok = load("others_base_s40000"), load("others_post_k3_s40000")
with open(os.path.join(OUT, "fresh_seeds_1000.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "group", "seed", "episode", "horizon", "base_success", "base_steps", "post_k3_success", "post_k3_steps", "identical_initial_state", "patch_blur_success", "patch_blur_steps"])
    for (b, e, g, s, key) in ((rb, rk, "test22", 30000, "rep_base_s30000|rep_post_k3_s30000"), (ob, ok, "other28", 40000, "others_base_s40000|others_post_k3_s40000")):
        for k in sorted(b):
            x, y = b[k], e[k]
            v = rv.get(k) if g == "test22" else None
            w.writerow([k[0], g, s + k[1], k[1], x["horizon"], int(x["success"]), x["steps"], int(y["success"]), y["steps"], int(is_exact(key, k[0], k[1])), "" if v is None else int(v["success"]), "" if v is None else v["steps"]])

# test set
bh, bs = load("base_heldout_s7000"), load("base_seen_recheck_s7000")
v1, v2, v3 = load("test_pblur06_s7000"), load("test_post_k3_s7000"), load("test_post_k5_s7000")
with open(os.path.join(OUT, "test_set_660.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "group", "seed", "episode", "horizon", "base_success", "base_steps", "patch_blur_success", "patch_blur_steps", "post_k3_success", "post_k3_steps", "post_k5_success", "post_k5_steps", "identical_initial_state_post_k3"])
    for (b, g, key) in ((bs, "seen", "base_seen_recheck_s7000|test_post_k3_s7000"), (bh, "held_out", "base_heldout_s7000|test_post_k3_s7000")):
        for k in sorted(b):
            x = b[k]
            w.writerow([k[0], g, 7000 + k[1], k[1], x["horizon"], int(x["success"]), x["steps"], int(v1[k]["success"]), v1[k]["steps"], int(v2[k]["success"]), v2[k]["steps"], int(v3[k]["success"]), v3[k]["steps"], int(is_exact(key, k[0], k[1]))])

# practice runs
DESC = {
    "skip5_p16_s20000": ("no", "Skip the first 5 steps of each chunk"),
    "jpeg90_p16_s20000": ("no", "JPEG quality 90 on the input"),
    "blur05_p16_s20000": ("no", "Image blur, sigma 0.5"),
    "blur08_p16_s20000": ("no", "Image blur, sigma 0.8"),
    "pblur06_p16_s20000": ("yes", "Patch embedding blur, sigma 0.6 (v1_patch_blur)"),
    "x264crf23_p16_s20000": ("no", "x264 CRF 23 stream encode and decode in the loop"),
    "s1_post_k5": ("yes", "Projection layer, input side: remove 5 directions (post_k5)"),
    "s1_post_k3": ("yes", "Projection layer, input side: remove 3 directions (post_k3)"),
    "s1_projablmean1": ("yes", "Projection layer, output side: remove 1 direction and shift bias"),
    "s1_projk1_a2": ("yes", "Projection layer, output side: 1 direction, alpha 2"),
    "s1_bias_m6": ("yes", "Projection bias shift only, c = -6"),
    "s1_bias_m4": ("yes", "Projection bias shift only, c = -4"),
    "s1_dct6g05": ("yes", "Patch embedding: halve high frequencies"),
    "s1_pblur07": ("yes", "Patch embedding blur, sigma 0.7"),
    "s1_pblur05": ("yes", "Patch embedding blur, sigma 0.5"),
    "s1_patchabl10": ("yes", "Patch embedding: remove 10 directions"),
    "s1_sig26_a6": ("yes", "SigLIP layer 26: remove direction, alpha 6"),
    "s1_projabl1": ("yes", "Projection layer, output side: remove 1 direction"),
    "s1_bb12_am6": ("yes", "Backbone layer 12: amplify direction, alpha -6"),
    "s1_bb12_am10": ("yes", "Backbone layer 12: amplify direction, alpha -10"),
    "s1_rope5": ("yes", "Action expert RoPE shift, delta 5"),
    "s1_exoL17_g13": ("yes", "Action expert layer 17 attention write-back x1.3"),
    "s1_randfilter06": ("yes", "Control: random filter with the spectrum of blur 0.6"),
    "s1_mit_recipe": ("yes", "Control: MIT recipe on action expert layers 12 and 15 (failure - success direction, top-64 columns, alpha 6)"),
    "s1_post_randgain0": ("yes", "Control: 3 random directions with matched gain"),
    "s1_post_tokpc": ("yes", "Control: 3 directions with the most token energy"),
    "s1_blurwrist08": ("no", "Blur the wrist camera only, sigma 0.8"),
    "s1_gain13": ("no", "Action gain 1.3"),
    "s1_cropleft95": ("no", "Left camera crop 0.95"),
    "s1_P_attnwrist1": ("yes", "Patch blur + wrist attention offset"),
    "s1_P_gain13": ("yes", "Patch blur + action gain 1.3"),
    "s1_P_steps20": ("yes", "Patch blur + 20 denoising steps"),
    "s1_P_attntask1": ("yes", "Patch blur + instruction attention offset"),
    "s1_P_exec3": ("yes", "Patch blur + execute 3 steps"),
    "s1_P_samples4": ("yes", "Patch blur + 4 samples"),
    "s1_P_exec8b": ("yes", "Patch blur + execute 8 steps"),
    "s1_P_cfg15": ("yes", "Patch blur + CFG 1.5"),
    "s1_P_wrist1_gain13": ("yes", "Patch blur + wrist attention + gain 1.3"),
    "s1_P_noise07": ("yes", "Patch blur + initial noise x0.7"),
    "s1_P_attnstate-1": ("yes", "Patch blur + state attention offset -1"),
    "s1_K3_pblur04": ("yes", "post_k3 + patch blur 0.4"),
    "s1_K3_modebias1": ("yes", "post_k3 + mode-bit bias"),
    "s1_K_cropleft95": ("yes", "post_k5 + left camera crop"),
    "s1_K_wrist1": ("yes", "post_k5 + wrist attention offset"),
    "s1_K_te3": ("yes", "post_k5 + temporal averaging over 3 chunks"),
    "s1_K_steps20": ("yes", "post_k5 + 20 denoising steps"),
    "s1_K_statebase": ("yes", "post_k5 + constant base pose in the state"),
    "s1_K_gain13": ("yes", "post_k5 + action gain 1.3"),
}
base = load("base_p16_s20000")
with open(os.path.join(OUT, "practice_runs.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "changes_weights", "description", "episodes", "base_success", "success", "more", "fewer", "mcnemar_p"])
    for name, (mw, desc) in DESC.items():
        p = os.path.join(F, name, "results.jsonl")
        if not os.path.exists(p):
            continue
        e = load(name)
        if name in ("s1_post_randgain0", "s1_post_tokpc") and len(e) < 80:
            continue
        keys = [k for k in base if k in e]
        up = sum(bool(e[k]["success"]) and not bool(base[k]["success"]) for k in keys)
        dn = sum(bool(base[k]["success"]) and not bool(e[k]["success"]) for k in keys)
        w.writerow([name, mw, desc, len(keys), sum(bool(base[k]["success"]) for k in keys), sum(bool(e[k]["success"]) for k in keys), up, dn, "%.3g" % mc(up, dn)])

# official check
O = os.path.join(EXP, "wsp_results", "official")
with open(os.path.join(OUT, "official_stack_check.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "official_success", "official_episodes", "ours_fresh_success", "ours_fresh_episodes", "ours_s7000_success", "ours_s7000_episodes"])
    tb = {**bh, **bs}
    fb = {**rb, **ob}
    for t in sorted(os.listdir(O)):
        p = os.path.join(O, t, "stats.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        a = [v["success"] for k, v in fb.items() if k[0] == t]
        c = [v["success"] for k, v in tb.items() if k[0] == t]
        w.writerow([t, round(s["success_rate"] * s["num_episodes"]), s["num_episodes"], sum(map(bool, a)), len(a), sum(map(bool, c)), len(c)])

# figure 1: speed by window, with task-level bootstrap
z = np.load(os.path.join(EXP, "wsp_results", "fig", "fig_data.npz"), allow_pickle=True)
tasks = z["pair_task"]
ut = sorted(set(tasks.tolist()))
idx = {t: np.where(tasks == t)[0] for t in ut}
rng = np.random.RandomState(0)
WIN = ["0-5", "5-10", "10-20", "20-35", "35-50"]
with open(os.path.join(OUT, "fig1_speed_by_window.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "window", "mean", "ci_low", "ci_high", "states", "tasks"])
    draws = [np.concatenate([idx[t] for t in rng.choice(ut, len(ut))]) for _ in range(2000)]
    for key, lab in (("sp_human", "human"), ("sp_mp4", "training_frame"), ("sp_edit", "raw_after_edit"), ("sp_raw", "raw_before_edit")):
        a = z[key]
        boot = np.array([a[s].mean(0) for s in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for i, x in enumerate(WIN):
            w.writerow([lab, x, round(float(a[:, i].mean()), 4), round(float(lo[i]), 4), round(float(hi[i]), 4), a.shape[0], len(ut)])


# offline dose tables cited in the post
def dose(name):
    rows = []
    for l in open(os.path.join(EXP, "wsp_results", "read", name)):
        m = re.match(r"^(\S+?)(?:（基座）)?\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) \| ([\d.]+) \| ([\d.]+)(?: \| rel_frob ([\d.]+))?", l)
        if m:
            rows.append(m.groups())
    return rows


with open(os.path.join(OUT, "offline_controls.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "edit", "speed_0_5", "speed_5_10", "speed_10_20", "speed_20_35", "speed_35_50", "dist_to_human_20steps", "dist_to_human_5steps", "action_diff_to_training_frame", "token_distance", "relative_frobenius_change"])
    for t in ("dose_pairs_d13.txt", "dose_pairs_d14.txt"):
        for r in dose(t):
            w.writerow([t.replace("dose_pairs_", "").replace(".txt", "")] + list(r))
for n in sorted(os.listdir(OUT)):
    print(n, sum(1 for _ in open(os.path.join(OUT, n))) - 1)
