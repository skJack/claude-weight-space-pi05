"""博客用的改前 / 改后并排视频：从两局录像存的逐步仿真状态重新渲染（只读录像，不改任何结果）。
左 = base，右 = post_k3，同任务同种子，两局录像的初始仿真状态逐位相同（断言）；每局都和录像存的画面比 PSNR，对不上就不出。主画面是模型的左相机，右上角小窗是手腕相机。3 倍速。
用法（envs/sim，cd /tmp）：MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<n> python -O render_clip.py <任务> <集合: rep|others> <局号> <输出目录> <输出名>
"""
import json, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from openpi_client import image_tools
import gymnasium as gym
import robocasa  # noqa
import robocasa.wrappers.gym_wrapper  # noqa

FWD = "/path/to/work/fwd"
DIRS = {"rep": ("rep_base_s30000", "rep_post_k3_s30000"), "others": ("others_base_s40000", "others_post_k3_s40000")}
FONT = "/path/to/work/blog/DejaVuSans-Bold.ttf"
N, IN = 512, 150
STRIDE, FPS, HOLD = 2, 30, 45
ORANGE, BLUE, GREEN, RED = (235, 104, 52), (42, 120, 214), (27, 175, 122), (227, 73, 72)


def unwrap(e):
    while not hasattr(e, "sim"):
        e = e.env if hasattr(e, "env") else e.unwrapped
    return e


def render_ep(path):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    S = z["sim_state"].astype(np.float64)
    qt = z["q_t"]
    rec = lambda i: cv2.imdecode(z["q_left"][i], cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)

    def check(k, i):
        k.sim.set_state_from_flattened(S[int(qt[i])])
        k.sim.forward()
        r = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(k.sim.render(camera_name="robot0_agentview_left", width=256, height=256)[::-1]), 224, 224))
        e = ((r.astype(np.float32) - rec(i)) ** 2).mean()
        return float(10 * np.log10(255 ** 2 / max(e, 1e-9)))

    for attempt in range(25):
        env = gym.make("robocasa/%s" % m["task"], split="target", seed=int(m["seed"]))
        env.reset()
        k = unwrap(env)
        try:
            p0 = check(k, 0)
        except ValueError:                                   # 状态维度都不一样，肯定不是同一个场景
            p0 = 0.0
        if p0 >= 35:
            break
        print("场景和录像不符，重建环境", m["name"], attempt, round(p0, 1), flush=True)
        env.close()
    else:
        raise SystemExit("重新渲染和录像对不上：%s" % m["name"])
    m["render_psnr"] = [round(check(k, int(i)), 1) for i in np.linspace(0, len(qt) - 1, 5)]
    fr = []
    idx = list(range(0, len(S), STRIDE))
    if idx[-1] != len(S) - 1:
        idx.append(len(S) - 1)
    for t in idx:
        k.sim.set_state_from_flattened(S[t])
        k.sim.forward()
        a = k.sim.render(camera_name="robot0_agentview_left", width=N, height=N)[::-1]
        w = k.sim.render(camera_name="robot0_eye_in_hand", width=IN, height=IN)[::-1]
        a = a.copy()
        a[10 - 2:10 + IN + 2, N - IN - 10 - 2:N - 10 + 2] = 255
        a[10:10 + IN, N - IN - 10:N - 10] = w
        fr.append(a)
    env.close()
    return m, idx, fr


def pill(d, xy, text, font, fill):
    x, y = xy
    b = d.textbbox((0, 0), text, font=font)
    w, h = b[2] - b[0], b[3] - b[1]
    d.rounded_rectangle((x, y, x + w + 20, y + h + 14), radius=7, fill=fill)
    d.text((x + 10, y + 7 - b[1]), text, font=font, fill=(255, 255, 255))


def compose(img, label, col, step, end_step, success, final, fonts):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    pill(d, (10, 10), label, fonts[0], col + (235,))
    if final:
        s = "%.1f s" % (end_step / 20.0)
        if success:
            pill(d, (10, N - 44), "✓ SUCCESS  " + s, fonts[0], GREEN + (240,))
        else:
            pill(d, (10, N - 44), "✗ TIME LIMIT  " + s, fonts[0], RED + (240,))
        c = GREEN if success else RED
        d.rectangle((0, 0, N - 1, N - 1), outline=c + (255,), width=5)
    else:
        pill(d, (10, N - 44), "%.1f s" % (step / 20.0), fonts[1], (0, 0, 0, 170))
    return np.asarray(im)


def main():
    task, which, ep, out, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    os.makedirs(out, exist_ok=True)
    db, de = DIRS[which]
    mb, ib, fb = render_ep(os.path.join(FWD, db, "episodes", "%s_base_ep%d.npz" % (task, ep)))
    me, ie, fe = render_ep(os.path.join(FWD, de, "episodes", "%s_base_ep%d.npz" % (task, ep)))
    assert mb["seed"] == me["seed"] and mb["layout_id"] == me["layout_id"] and mb["style_id"] == me["style_id"]
    s0 = np.load(os.path.join(FWD, db, "episodes", "%s_base_ep%d.npz" % (task, ep)), allow_pickle=True)["sim_state"][0]
    s1 = np.load(os.path.join(FWD, de, "episodes", "%s_base_ep%d.npz" % (task, ep)), allow_pickle=True)["sim_state"][0]
    assert (s0 == s1).all(), "两局的初始状态不同"
    fonts = (ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 18))
    L = max(len(fb), len(fe))
    W, H = 2 * N + 8, N
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(out, name + ".mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    gap = np.full((H, 8, 3), 255, np.uint8)
    strip_at = [int(x) for x in np.linspace(0, L - 1, 6)]
    strip = {}
    for i in range(L + HOLD):
        j = min(i, L - 1)
        fb_i, fe_i = min(j, len(fb) - 1), min(j, len(fe) - 1)
        A = compose(fb[fb_i], "BEFORE", ORANGE, ib[fb_i], mb["steps"], bool(mb["success"]), j >= len(fb) - 1, fonts)
        B = compose(fe[fe_i], "AFTER", BLUE, ie[fe_i], me["steps"], bool(me["success"]), j >= len(fe) - 1, fonts)
        frame = np.hstack([A, gap, B])
        p.stdin.write(frame.tobytes())
        if i == 0:
            Image.fromarray(frame).save(os.path.join(out, name + "_poster.jpg"), quality=82)
        if i in strip_at:
            strip[i] = frame
    p.stdin.close()
    p.wait()
    Image.fromarray(np.vstack([strip[i] for i in strip_at])[::2, ::2]).save(os.path.join(out, name + "_sheet.jpg"), quality=70)
    # 博客的胶片条：按固定时刻取两边的帧，存原始尺寸
    fs = {}
    for sec in [int(x) for x in os.environ.get("FILM_SECS", "0,5,10,15,20,30").split(",")]:
        q = int(sec * 20 / STRIDE)
        fs["b%d" % sec] = fb[min(q, len(fb) - 1)]
        fs["e%d" % sec] = fe[min(q, len(fe) - 1)]
    np.savez_compressed(os.path.join(out, name + "_film.npz"), **fs)
    info = dict(name=name, task=task, ep=ep, seed=mb["seed"], lang=mb["lang"], layout=mb["layout_id"], style=mb["style_id"],
                base_success=bool(mb["success"]), base_steps=int(mb["steps"]), edit_success=bool(me["success"]), edit_steps=int(me["steps"]),
                base_render_psnr=mb["render_psnr"], edit_render_psnr=me["render_psnr"])
    json.dump(info, open(os.path.join(out, name + ".json"), "w"), ensure_ascii=False)
    print("OK", info, flush=True)


if __name__ == "__main__":
    main()
