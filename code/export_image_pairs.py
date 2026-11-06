"""博客用的图像对比：同一仿真状态的渲染原图和训练视频帧（H.264 解码）。只从 pairs2_B（没参与估计方向的任务）里取。
先出一张候选总览，挑好了再导出单张。
用法：python blog_pairs.py sheet <out.jpg>
      python blog_pairs.py export <下标> <相机> <out_dir>
"""
import sys, os
import numpy as np, cv2
z = np.load("/path/to/work/read/pairs2_B.npz", allow_pickle=True)
N = len(z["t"])


def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]


if sys.argv[1] == "sheet":
    rows = []
    tasks = sorted(set(z["task"].tolist()))
    pick = []
    for t in tasks:
        ids = np.where(z["task"] == t)[0]
        pick.append(int(ids[len(ids) // 2]))
    for i in pick[:27]:
        tiles = []
        for c in ("left", "wrist", "right"):
            r, m = dec(z["raw_" + c][i]), dec(z["mp4_" + c][i])
            d = np.clip(np.abs(r.astype(float) - m.astype(float)) * 8, 0, 255).astype(np.uint8)
            tiles += [r, m, d]
        row = np.hstack(tiles)
        row = cv2.resize(row, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).copy()
        cv2.putText(row, "%d %s" % (i, z["task"][i]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)
    half = len(rows) // 2
    cv2.imwrite(sys.argv[2], np.vstack(rows[:half])[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(sys.argv[2].replace(".jpg", "_2.jpg"), np.vstack(rows[half:])[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
else:
    i, c, out = int(sys.argv[2]), sys.argv[3], sys.argv[4]
    os.makedirs(out, exist_ok=True)
    r, m = dec(z["raw_" + c][i]), dec(z["mp4_" + c][i])
    n = "%s_%d_%s" % (z["task"][i], i, c)
    cv2.imwrite(os.path.join(out, n + "_raw.png"), r[:, :, ::-1])
    cv2.imwrite(os.path.join(out, n + "_mp4.png"), m[:, :, ::-1])
    mse = ((r.astype(float) - m.astype(float)) ** 2).mean()
    print(n, "PSNR %.1f dB" % (10 * np.log10(255 ** 2 / mse)), "lang:", z["lang"][i])
