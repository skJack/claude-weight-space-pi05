"""Build index.html (English) and zh.html (Chinese) from en.html, zh.html and the CSV files in results/.
Usage: python3 site_src/build.py   (from the repository root). Standard library only."""
import csv, html, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AUTHOR = "Jack Sun Kelip"
REPO = "https://github.com/skJack/claude-weight-space-pi05"
SITE = "https://skjack.github.io/claude-weight-space-pi05/"


def rows(name):
    return list(csv.DictReader(open(os.path.join(ROOT, "results", name))))


T = {
    "en": {
        "lang": "en", "file": "index.html", "other": "zh.html", "other_label": "中文", "this_label": "EN", "code": "Code and data",
        "title": "Claude Programs a Robot in Weight Space",
        "desc": "No gradients and no training. Claude changed one matrix inside π0.5, the model driving the robot, and success on 50 RoboCasa tasks rose from 51.9% to 65.1%.",
        "before": "Before", "after": "After",
        "video_helped": "Original failed, edited succeeded", "video_other": "Where it didn't help",
        "video_cap": "Same task and same starting state, with one matrix changed. Left: original. Right: edited. Main view: the model's left camera. Corner: wrist camera. 3× speed. Clips are re-rendered at higher resolution from the recorded simulator states and checked against the original recordings.",
        "meta": "Before: {b}. After: {a}. Seed {seed}. This task overall: {bt} → {at} of 20.",
        "ok": "success at {s:.1f} s", "fail": "time limit at {s:.1f} s",
        "film_title": "Frames from the same two episodes", "sec": "{} s",
        "film_cap": "The same episode pair as the mug clip above, at the same moments. The edited run finishes at {s:.1f} s, and its later frames stay on the final state. The original is still at the machine when time runs out at 33.8 s.",
        "raw": "Raw render", "mp4": "Training frame", "diff": "Difference × 8", "zoom": "Zoom × 4", "drag": "Drag to compare",
        "pair_cap": "{name}. Wrist camera, same simulator state, not used to estimate the edit. PSNR {psnr} dB. Left of the divider is what the model sees in evaluation, and right of it is what it saw in training. The difference panel shows the absolute difference, amplified 8 times.",
        "pair_names": {"potato": "Potato", "onion": "Onion", "hotdog": "Hot dog"},
        "fig1_title": "Figure 1. Fed a sharp render, the model plans a slow start. The edit brings it back.",
        "fig1_series": {"human": "Human demonstration", "train": "Training frame", "after": "Render, after edit", "before": "Render, before edit"},
        "fig1_x": "Step in the 50-step action chunk", "fig1_y": "Translation per step (action units)",
        "fig1_d_before": "Render, before edit", "fig1_d_after": "After edit ≈ training frame", "fig1_d_human": "Human",
        "fig1_note": "Only this part runs before the next replan",
        "fig1_cap": "305 states from 27 tasks that were not used to estimate the edit, with the same state and the same noise for every line. For the first 5 steps, after minus before is +0.199 (95% CI 0.18 to 0.22), and training frame minus after is +0.009 (95% CI −0.002 to 0.023). The intervals come from resampling tasks.",
        "data": "Show the numbers", "window": "Steps",
        "fig2_title": "Figure 2. Success per task, 20 paired episodes each",
        "fig2_groups": ("{} tasks went up", "{} tasks unchanged", "{} tasks went down"),
        "fig2_cap": "Fresh seeds. 35 tasks went up, 4 stayed the same and 11 went down. With 20 episodes per task, single-task differences can be noise.",
        "controls_pending": "",
        "controls": "In closed loop, over 80 practice episodes where the original model gets 20, the random directions get {r} and the high-energy directions get {t}, against 39 for this edit.",
    },
    "zh": {
        "lang": "zh-CN", "file": "zh.html", "other": "index.html", "other_label": "EN", "this_label": "中文", "code": "代码和数据",
        "title": "Claude 在权重空间里给机器人编程",
        "desc": "不用梯度，不训练。Claude 改了驱动机器人的模型 π0.5 里的一个矩阵，50 个 RoboCasa 任务的成功率从 51.9% 升到 65.1%。",
        "before": "改前", "after": "改后",
        "video_helped": "改前失败，改后成功", "video_other": "没有帮上忙的",
        "video_cap": "同一任务、同一初始状态，只改了一个矩阵。左边是改前，右边是改后。主画面是模型的左相机，角上是手腕相机，3 倍速。片段是用录像存的仿真状态重新渲染的高清版，并且和原始录像核对过。",
        "meta": "改前：{b}。改后：{a}。种子 {seed}。这个任务整体：20 局里 {bt} → {at}。",
        "ok": "{s:.1f} 秒成功", "fail": "{s:.1f} 秒到时限",
        "film_title": "同一对局在相同时刻的画面", "sec": "{} 秒",
        "film_cap": "和上面取杯子的片段是同一对局。改后在 {s:.1f} 秒完成，之后的画面停在最后状态。改前到 33.8 秒时限还停在咖啡机前。",
        "raw": "渲染原图", "mp4": "训练视频帧", "diff": "差异 × 8", "zoom": "放大 4 倍", "drag": "拖动比较",
        "pair_cap": "{name}。手腕相机，同一仿真状态，没有用于估计改动。PSNR {psnr} dB。分割线左边是模型评测时看到的，右边是它训练时看到的。差异图是绝对差异放大 8 倍的结果。",
        "pair_names": {"potato": "土豆", "onion": "洋葱", "hotdog": "热狗"},
        "fig1_title": "图 1. 喂清晰的渲染图，模型把开头规划得很慢。改动把它拉了回来。",
        "fig1_series": {"human": "人的示教", "train": "训练视频帧", "after": "渲染图，改后", "before": "渲染图，改前"},
        "fig1_x": "50 步 action chunk 里的第几步", "fig1_y": "每步平移（动作空间单位）",
        "fig1_d_before": "渲染图，改前", "fig1_d_after": "改后 ≈ 训练视频帧", "fig1_d_human": "人",
        "fig1_note": "下一次重规划前只执行这一段",
        "fig1_cap": "305 个状态，来自 27 个没有用于估计改动的任务，每条线都用同一状态、同一噪声。前 5 步里，改后减改前是 +0.199（95% CI 0.18 到 0.22），训练帧减改后是 +0.009（95% CI −0.002 到 0.023）。区间按任务重采样得到。",
        "data": "看具体数字", "window": "步",
        "fig2_title": "图 2. 每个任务的成功局数，各 20 对局",
        "fig2_groups": ("{} 个任务上升", "{} 个任务不变", "{} 个任务下降"),
        "fig2_cap": "新种子。35 个任务上升，4 个不变，11 个下降。每任务只有 20 局，单个任务的差异可能是噪声。",
        "controls_pending": "",
        "controls": "闭环上，80 个练习局里未改模型成功 20 局，随机方向 {r} 局，能量最大的方向 {t} 局，这个改动 39 局。",
    },
}

CLIPS = [
    ("coffee", "helped", "Take the mug out", "把杯子取出来"),
    ("navigate", "helped", "Go to the microwave", "走到微波炉"),
    ("kettle", "helped", "Kettle lever", "按水壶开关"),
    ("toaster", "helped", "Close the toaster oven", "关小烤箱门"),
    ("blender", "helped", "Orange into the blender", "橙子放进搅拌机"),
    ("microwave", "helped", "Eggplant out of the microwave", "把茄子从微波炉里取出来"),
    ("worse_blender", "other", "Got worse: blender button", "变差：搅拌机按钮"),
    ("worse_toaster", "other", "Got worse: bread into toaster oven", "变差：面包放进小烤箱"),
    ("both_oven", "other", "Neither: open the oven", "都失败：开烤箱门"),
]

PAIRS = [("potato", 40.4, 56, 56), ("onion", 43.4, 50, 82), ("hotdog", 35.1, 82, 50)]


def task_counts():
    c = {}
    for r in rows("fresh_seeds_1000.csv"):
        a = c.setdefault(r["task"], [0, 0, 0])
        a[0] += int(r["base_success"])
        a[1] += int(r["post_k3_success"])
        a[2] += 1
    return c


COUNTS = task_counts()


def gallery(L):
    t = T[L]
    buttons = {"helped": [], "other": []}
    first = None
    for name, group, en, zh in CLIPS:
        info = json.load(open(os.path.join(ROOT, "assets", "video", name + ".json")))
        fmt = lambda ok, st: (t["ok"] if ok else t["fail"]).format(s=st / 20.0)
        bt, at, _ = COUNTS[info["task"]]
        meta = t["meta"].format(b=fmt(info["base_success"], info["base_steps"]), a=fmt(info["edit_success"], info["edit_steps"]), seed=info["seed"], bt=bt, at=at)
        d = dict(src="assets/video/%s.mp4" % name, poster="assets/video/%s_poster.jpg" % name, prompt="“%s”" % info["lang"], meta=meta)
        if first is None:
            first = d
        attrs = " ".join('data-%s="%s"' % (k, html.escape(v, quote=True)) for k, v in d.items())
        buttons[group].append('<button type="button" class="tab" aria-pressed="%s" %s>%s</button>' % ("true" if d is first else "false", attrs, html.escape(en if L == "en" else zh)))
    return """<figure class="wide gallery" data-gallery>
  <div class="tabs-group"><span class="tabs-label">%s</span><div class="tabs">%s</div></div>
  <div class="tabs-group"><span class="tabs-label">%s</span><div class="tabs">%s</div></div>
  <div class="video-frame"><video src="%s" poster="%s" muted loop playsinline controls preload="metadata"></video></div>
  <figcaption><p class="clip-prompt">%s</p><p class="clip-meta">%s</p><p style="margin:.5rem 0 0">%s</p></figcaption>
</figure>""" % (t["video_helped"], "".join(buttons["helped"]), t["video_other"], "".join(buttons["other"]), first["src"], first["poster"], html.escape(first["prompt"]), html.escape(first["meta"]), t["video_cap"])


def film(L):
    t = T[L]
    times = (0, 2, 4, 6, 8, 20)
    info = json.load(open(os.path.join(ROOT, "assets", "video", "coffee.json")))
    head = '<span></span>' + "".join('<span class="film-time">%s</span>' % t["sec"].format(s) for s in times)
    row = lambda who, lab, cls: '<span class="film-row"><i class="key %s"></i>%s</span>' % (cls, lab) + "".join('<img src="assets/img/film/%s%d.jpg" alt="%s, %s" loading="lazy">' % (who, s, lab, t["sec"].format(s)) for s in times)
    return """<figure class="wide">
  <p class="chart-title">%s</p>
  <div class="film"><div class="film-grid">%s%s%s</div></div>
  <figcaption>%s</figcaption>
</figure>""" % (t["film_title"], head, row("b", t["before"], "before"), row("e", t["after"], "after"), t["film_cap"].format(s=info["edit_steps"] / 20.0))


def compare(L):
    t = T[L]
    btn = []
    for i, (k, psnr, x, y) in enumerate(PAIRS):
        cap = t["pair_cap"].format(name=t["pair_names"][k], psnr=psnr)
        btn.append('<button type="button" class="tab" aria-pressed="%s" data-raw="assets/img/pairs/%s_raw.png" data-mp4="assets/img/pairs/%s_mp4.png" data-diff="assets/img/pairs/%s_diff.png" data-x="%d" data-y="%d" data-caption="%s">%s</button>' % ("true" if i == 0 else "false", k, k, k, x, y, html.escape(cap, quote=True), t["pair_names"][k]))
    k0 = PAIRS[0][0]
    return """<figure class="wide" data-compare>
  <div class="sample-tabs">%s<label class="zoom-toggle"><input type="checkbox">%s</label></div>
  <div class="compare-grid">
    <div class="stage slider">
      <div class="zoomer"><img class="base" src="assets/img/pairs/%s_mp4.png" alt="%s"><img class="top" src="assets/img/pairs/%s_raw.png" alt="%s"></div>
      <div class="divider"></div>
      <input type="range" min="0" max="100" value="50" aria-label="%s">
      <span class="tag l">%s</span><span class="tag r">%s</span>
    </div>
    <div class="stage">
      <div class="zoomer"><img class="diff" src="assets/img/pairs/%s_diff.png" alt="%s"></div>
      <span class="tag l">%s</span>
    </div>
  </div>
  <figcaption class="pair-caption">%s</figcaption>
</figure>""" % ("".join(btn), t["zoom"], k0, t["mp4"], k0, t["raw"], t["drag"], t["raw"], t["mp4"], k0, t["diff"], t["diff"], t["pair_cap"].format(name=t["pair_names"][k0], psnr=PAIRS[0][1]))


def fig_speed(L):
    t = T[L]
    data = {}
    for r in rows("fig1_speed_by_window.csv"):
        data.setdefault(r["series"], []).append(float(r["mean"]))
    mids = [2.5, 7.5, 15, 27.5, 42.5]
    series = (("human", "human", "var(--human)"), ("training_frame", "train", "var(--train)"), ("raw_before_edit", "before", "var(--before)"), ("raw_after_edit", "after", "var(--after)"))
    labels = t["fig1_series"]
    windows = ["0–5", "5–10", "10–20", "20–35", "35–50"]

    def draw(W, H, x0, x1, y0, y1, xticks, cls):
        ymin, ymax = 0.28, 0.70
        X = lambda v: x0 + (x1 - x0) * v / 50.0
        Y = lambda v: y0 + (y1 - y0) * (v - ymin) / (ymax - ymin)
        s = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s">' % (W, H, html.escape(t["fig1_title"], quote=True))]
        s.append('<rect x="%.1f" y="%d" width="%.1f" height="%d" fill="var(--wash)"/>' % (X(0), y1, X(5) - X(0), y0 - y1))
        for v in (0.3, 0.4, 0.5, 0.6, 0.7):
            s.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="var(--grid)" stroke-width="1"/>' % (x0, x1, Y(v), Y(v)))
            s.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%.1f</text>' % (x0 - 8, Y(v) + 4, v))
        s.append('<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="var(--axis)" stroke-width="1"/>' % (x0, x1, y0, y0))
        for v in xticks:
            s.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%d</text>' % (X(v), y0 + 20, v))
        s.append('<text class="axis-label" x="%.1f" y="%d" text-anchor="middle">%s</text>' % ((x0 + x1) / 2, y0 + 43, t["fig1_x"]))
        s.append('<text class="axis-label" transform="translate(15 %.1f) rotate(-90)" text-anchor="middle">%s</text>' % ((y0 + y1) / 2, t["fig1_y"]))
        for key, c, col in series:
            s.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>' % (" ".join("%.1f,%.1f" % (X(m), Y(a)) for m, a in zip(mids, data[key])), col))
        for key, c, col in series:
            for m, a in zip(mids, data[key]):
                s.append('<circle cx="%.1f" cy="%.1f" r="4.4" fill="%s" stroke="var(--surface)" stroke-width="2"/>' % (X(m), Y(a), col))
        s.append('<text class="direct" x="%.1f" y="%.1f">%s</text>' % (X(10.5), Y(0.415), t["fig1_d_before"]))
        if cls == "big":
            s.append('<text class="direct" x="%.1f" y="%.1f">%s</text>' % (X(10.5), Y(0.546), t["fig1_d_after"]))
        s.append('<text class="direct" x="%.1f" y="%.1f">%s</text>' % (X(8.2), Y(0.668), t["fig1_d_human"]))
        s.append('<line x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f" stroke="var(--muted)" stroke-width="1"/>' % (X(2.5), X(6.5), Y(0.302), Y(0.302)))
        s.append('<text class="note" x="%.1f" y="%.1f">%s</text>' % (X(7), Y(0.302) + 4, t["fig1_note"]))
        s.append('<line class="guide" x1="0" x2="0" y1="%d" y2="%d" stroke="var(--muted)" stroke-width="1" opacity="0"/>' % (y1, y0))
        s.append("</svg>")
        win = [w + " " + t["window"] if L == "zh" else t["window"] + " " + w for w in windows]
        payload = dict(vbw=W, x0=x0, x1=x1, xs=[round(X(m), 1) for m in mids], windows=win, series=[dict(key=c, label=labels[c], values=data[k]) for k, c, _ in series])
        return """<div class="linechart %s" data-linechart='%s'>%s<div class="tooltip"></div></div>""" % (cls, html.escape(json.dumps(payload, ensure_ascii=False), quote=True).replace("&#x27;", "'"), "".join(s))

    legend = "".join('<span><i class="key %s"></i>%s</span>' % (c, labels[c]) for _, c, _ in series)
    tr = "".join("<tr><td>%s</td>%s</tr>" % (labels[c], "".join("<td>%.3f</td>" % a for a in data[k])) for k, c, _ in series)
    return """<figure class="wide card">
  <p class="chart-title">%s</p>
  <div class="legend">%s</div>
  %s%s
  <figcaption>%s</figcaption>
  <details><summary>%s</summary><div class="data-table"><table><thead><tr><th>%s</th>%s</tr></thead><tbody>%s</tbody></table></div></details>
</figure>""" % (t["fig1_title"], legend, draw(860, 430, 66, 846, 382, 16, (0, 5, 10, 20, 35, 50), "big"), draw(430, 400, 58, 420, 346, 12, (0, 10, 20, 35, 50), "small"), t["fig1_cap"], t["data"], t["window"], "".join("<th>%s</th>" % w for w in windows), tr)


def fig_tasks(L):
    t = T[L]
    items = sorted(COUNTS.items(), key=lambda kv: (-(kv[1][1] - kv[1][0]), kv[0]))
    groups = [[kv for kv in items if kv[1][1] > kv[1][0]], [kv for kv in items if kv[1][1] == kv[1][0]], [kv for kv in items if kv[1][1] < kv[1][0]]]
    pct = lambda v: 100.0 * v / 20
    lines = "".join('<div class="gridline" style="left:%g%%"></div>' % pct(v) for v in (0, 5, 10, 15, 20))
    out = ['<div class="task-head"><span></span><div class="task-axis">%s</div><span></span></div>' % "".join('<span style="left:%g%%">%d</span>' % (pct(v), v) for v in (0, 5, 10, 15, 20))]
    for g, label in zip(groups, t["fig2_groups"]):
        out.append('<div class="task-group">%s</div>' % label.format(len(g)))
        for name, (b, a, _) in g:
            lo, hi = sorted((b, a))
            seg = '<div class="seg" style="left:%g%%;width:%g%%"></div>' % (pct(lo), pct(hi - lo)) if hi > lo else ""
            out.append('<div class="task-row" title="%s: %d → %d / 20"><span class="task-name">%s</span><div class="track">%s%s<div class="dot before" style="left:%g%%"></div><div class="dot after" style="left:%g%%"></div></div><span class="task-val">%d → %d</span></div>' % (name, b, a, name, lines, seg, pct(b), pct(a), b, a))
    legend = '<div class="legend"><span><i class="key before"></i>%s</span><span><i class="key after"></i>%s</span></div>' % (t["before"], t["after"])
    return """<figure class="wide card">
  <p class="chart-title">%s</p>
  %s
  <div class="tasks">%s</div>
  <figcaption>%s</figcaption>
</figure>""" % (t["fig2_title"], legend, "".join(out), t["fig2_cap"])


def controls(L):
    t = T[L]
    got = {r["run"]: r for r in rows("practice_runs.csv")}
    if "s1_post_randgain0" in got and "s1_post_tokpc" in got:
        return t["controls"].format(r=got["s1_post_randgain0"]["success"], t=got["s1_post_tokpc"]["success"])
    return t["controls_pending"]


PAGE = """<!doctype html>
<html lang="%(lang)s">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<meta name="description" content="%(desc)s">
<meta property="og:type" content="article">
<meta property="og:title" content="%(title)s">
<meta property="og:description" content="%(desc)s">
<meta property="og:url" content="%(url)s">
<meta property="og:image" content="%(site)sassets/img/card.jpg">
<meta name="twitter:card" content="summary_large_image">
<link rel="alternate" hreflang="en" href="%(site)s">
<link rel="alternate" hreflang="zh" href="%(site)szh.html">
<link rel="icon" href="data:image/svg+xml,%%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%%3E%%3Crect width='32' height='32' rx='7' fill='%%232a78d6'/%%3E%%3Cpath d='M9 22V10l14 12V10' fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%%3E%%3C/svg%%3E">
<link rel="stylesheet" href="assets/site.css">
</head>
<body lang="%(lang)s">
<div class="topbar">
  <a href="%(repo)s">%(code)s</a>
  <nav><span class="lang-current">%(this_label)s</span><a href="%(other)s" hreflang="%(other_lang)s">%(other_label)s</a></nav>
</div>
<main>
%(body)s
</main>
<footer>
%(footer)s
</footer>
<script src="assets/site.js"></script>
</body>
</html>
"""

FOOT = {
    "en": """<p>π0.5 and openpi are by Physical Intelligence. The benchmark, dataset and checkpoint are by the RoboCasa team. The idea comes from <a href="https://rikiyat.github.io/blog/weight-space-programming/">Takehi, Teoh, Zhang and Khattab</a> at MIT CSAIL. Those groups were not involved in this work.</p>
<p>The experiments and this post's first draft were done by Claude Opus 5.5, with {author} directing and reviewing. If you find a mistake, please <a href="{repo}/issues">open an issue</a>.</p>""",
    "zh": """<p>π0.5 和 openpi 来自 Physical Intelligence，基准、数据集和权重来自 RoboCasa 团队，想法来自 MIT CSAIL 的 <a href="https://rikiyat.github.io/blog/weight-space-programming/">Takehi、Teoh、Zhang 和 Khattab</a>。这些团队没有参与这项工作。</p>
<p>实验和这篇文章的初稿由 Claude Opus 5.5 完成，{author} 负责指导和审阅。如果发现错误，欢迎<a href="{repo}/issues">提 issue</a>。</p>""",
}


def build(L):
    t = T[L]
    body = open(os.path.join(ROOT, "site_src", L + ".html")).read()
    for k, v in (("{{author}}", AUTHOR), ("{{repo}}", REPO), ("{{gallery}}", gallery(L)), ("{{film}}", film(L)), ("{{compare}}", compare(L)), ("{{fig_speed}}", fig_speed(L)), ("{{fig_tasks}}", fig_tasks(L)), ("{{closed_loop_controls}}", controls(L))):
        body = body.replace(k, v)
    assert "{{" not in body, body[body.index("{{"):][:40]
    out = PAGE % dict(lang=t["lang"], title=t["title"], desc=html.escape(t["desc"], quote=True), url=SITE + ("" if L == "en" else "zh.html"), site=SITE, repo=REPO, code=t["code"], this_label=t["this_label"], other=t["other"], other_label=t["other_label"], other_lang="zh" if L == "en" else "en", body=body, footer=FOOT[L].format(author=AUTHOR, repo=REPO))
    open(os.path.join(ROOT, t["file"]), "w").write(out)
    print("wrote", t["file"], len(out))


for L in ("en", "zh"):
    build(L)
