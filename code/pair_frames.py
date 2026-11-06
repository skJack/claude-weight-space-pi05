"""同一仿真状态的两张图：训练视频里的帧（H.264 解码）vs 现在闭环里渲染的帧。用来量"训练时看到的图"和"闭环里喂的图"差多少。
示教来自 π0.5 的训练集（pretrain split）：按 states.npz 恢复仿真状态，用和 gym wrapper 相同的方式渲染 256×256，再缩到 224。
用法（envs/sim，cd /tmp）：MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<n> python -O pair_frames.py <out.npz> [--tasks a,b] [--eps 2] [--every 30]
"""
import argparse, glob, gzip, json, os, sys
from pathlib import Path
import numpy as np

CAMS = (("left", "robot0_agentview_left"), ("wrist", "robot0_eye_in_hand"), ("right", "robot0_agentview_right"))
DATA = "/path/to/robocasa_root/robocasa/datasets/v1.0/pretrain/atomic"
H = 50


def png(img):
    import cv2
    ok, b = cv2.imencode(".png", img[:, :, ::-1])
    return np.frombuffer(b.tobytes(), np.uint8)


def r224(img):
    from openpi_client import image_tools
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(img), 224, 224))


def decode(ds, ep, cam):
    import av
    p = sorted(Path(ds).glob(f"videos/*/observation.images.{cam}/episode_{ep:06d}.mp4"))[0]
    with av.open(str(p)) as c:
        return [fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]


def main():
    import pandas as pd
    import robosuite
    import robocasa  # noqa: F401
    from robocasa.utils import lerobot_utils as LU
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tasks", default="TurnOnMicrowave,TurnOnStove,CloseToasterOvenDoor,PickPlaceCounterToOven,OpenDrawer,TurnOnBlender")
    ap.add_argument("--eps", type=int, default=2)
    ap.add_argument("--every", type=int, default=30)
    a = ap.parse_args()
    rows = {k: [] for k in ("task", "ep", "t", "lang", "state", "target")}
    for c, _ in CAMS:
        rows["raw_" + c], rows["mp4_" + c] = [], []
    for task in a.tasks.split(","):
        ds = sorted(glob.glob(os.path.join(DATA, task, "*", "lerobot")))[-1]
        meta = LU.get_env_metadata(Path(ds))
        kw = dict(meta["env_kwargs"])
        kw["env_name"] = meta["env_name"]
        kw.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False)
        env = robosuite.make(**kw)
        for ep in range(a.eps):
            ed = os.path.join(ds, "extras", "episode_%06d" % ep)
            states = np.load(os.path.join(ed, "states.npz"))["states"]
            with gzip.open(os.path.join(ed, "model.xml.gz"), "rt") as f:
                xml = f.read()
            ep_meta = json.load(open(os.path.join(ed, "ep_meta.json")))
            if hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
            try:
                env.reset(unset_ep_meta=False)
            except TypeError:
                env.reset()
            env.reset_from_xml_string(env.edit_model_xml(xml))
            env.sim.reset()
            df = pd.read_parquet(sorted(Path(ds).glob(f"data/*/episode_{ep:06d}.parquet"))[0])
            act = np.stack(df["action"].values).astype(np.float32)
            obs = np.stack(df["observation.state"].values).astype(np.float32)
            st = np.concatenate([obs[:, 7:10], obs[:, 10:14], obs[:, 0:3], obs[:, 3:7], obs[:, 14:16]], 1)   # LeRobot → π0.5 顺序
            ae = np.concatenate([act[:, 5:8], act[:, 8:11], act[:, 11:12], act[:, 0:4], act[:, 4:5]], 1)
            frames = {cam: decode(ds, ep, cam) for _, cam in CAMS}
            T = min(len(states), len(act))
            for t in range(10, T - 10, a.every):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                for c, cam in CAMS:
                    raw = env.sim.render(camera_name=cam, width=256, height=256)[::-1]
                    rows["raw_" + c].append(png(r224(raw)))
                    rows["mp4_" + c].append(png(r224(frames[cam][t])))
                ch = ae[t:t + H]
                rows["target"].append(np.concatenate([ch, np.repeat(ch[-1:], H - len(ch), 0)]) if len(ch) < H else ch)
                rows["state"].append(st[t])
                rows["task"].append(task)
                rows["ep"].append(ep)
                rows["t"].append(t)
                rows["lang"].append(ep_meta.get("lang", ""))
            print(task, ep, T, flush=True)
        env.close()
    arrs = {}
    for k, v in rows.items():
        if k.startswith("raw_") or k.startswith("mp4_"):
            o = np.empty(len(v), dtype=object)
            o[:] = v
            arrs[k] = o
        else:
            arrs[k] = np.asarray(v)
    np.savez(a.out, **arrs)
    print("PAIR FRAMES DONE", len(rows["t"]), a.out, flush=True)


if __name__ == "__main__":
    main()
