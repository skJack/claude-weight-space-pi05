"""权重改动：存成"对基座权重的精确 diff + 哈希"（照 MIT 实验 8 的 edit/manifest.json + apply_edit.py）。
一条改动 = 一个 npz，里面：
  ops：JSON，列表，每项 {"name": 参数名, "kind": ..., ...}
  u_<i>：方向（单位向量）
kind：
  ablate：W' = W − α·u·(mask ⊙ uᵀW)，W 形状 (d, n)，u 长度 d，mask 是选中的列（空 = 全部列）。α > 0 消掉，α < 0 放大。
  out_row：action_out_proj.weight[row, :] *= (1 + α)。
  rope_shift：π0.5 的动作 token 只靠 RoPE 知道自己是块里第几步。RoPE 是对 q / k 按位置 p 做的分块旋转 R(p)，R(p + Δ) = R(p)·R(Δ)，
    所以把 expert 的 q_proj、k_proj 左乘 R(Δ)，就和"动作 token 的 position_ids 全部 + Δ"等价（前缀的 key 来自骨干，不动）：
    第 i 个 token 算的是块里第 i + Δ 步。一个数 Δ，但 q_proj / k_proj 的每个元素都会变。op 里写 {"name": 参数名, "kind": "rope_shift", "delta": Δ}。
  patch_blur：SigLIP 的 patch embedding 是线性的，E(x) = W·x + b，x 是 14×14×3 的 patch。W' = W·G 就等于先对 patch 做一次高斯模糊（G 的行归一，σ 单位是像素；
    模糊只在 patch 内部，边界用不到相邻 patch 的像素）。op：{"name": patch_embedding.weight, "kind": "patch_blur", "sigma": σ}。这个张量运行时是 float32，不受 bfloat16 舍入影响。
  patch_matrix：W' = W·L，L 是存在 npz 里的 196×196（每个颜色通道共用）或 588×588 矩阵（键 L_<i>），可以再带偏移 m_<i>（b' = b + W·m）。
  bias_add：张量（偏置）加上存的向量 v_<i>。
  subspace_abl：W' = (I − UUᵀ)W，U 是存的正交列（U_<i>，d×k）。W 可以是 d×n，也可以是 patch embedding 的 (d, 3, 14, 14)。偏置不动（同时消偏置要另加一条 op）。
  input_abl：W' = W·(I − α·VVᵀ)，V 是输入空间的正交列（V_<i>，n×k）：在这一层的输入一侧消方向。
  scale：W' = g·W（op 里写 g）。比如放大一层注意力的 o_proj，就是放大这一层注意力写回残差流的量。
基座权重只读；改完在原 dtype（bfloat16）下写回，返回每个张量的改动规模和 sha256。
"""
import hashlib, json
import numpy as np
import torch


def _sha(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def rope_ops(layers, delta, which="qk"):
    P = "paligemma_with_expert.gemma_expert.model.layers.%d.self_attn.%s_proj.weight"
    return [{"name": P % (l, w), "kind": "rope_shift", "delta": float(delta), "layer": int(l)} for l in layers for w in which]


def apply_edit(model, path):
    z = np.load(path, allow_pickle=True)
    ops = json.loads(str(z["ops"]))
    params = dict(model.named_parameters())
    report = []
    with torch.no_grad():
        for i, op in enumerate(ops):
            p = params[op["name"]]
            W = p.data.float()
            before = W.clone()
            if op["kind"] == "ablate":
                u = torch.as_tensor(z["u_%d" % i], device=W.device, dtype=torch.float32)
                u = u / u.norm()
                coef = u @ W                                    # 每列在 u 上的分量
                if op.get("cols"):
                    mask = torch.zeros_like(coef)
                    mask[torch.as_tensor(op["cols"], device=W.device)] = 1.0
                    coef = coef * mask
                W = W - float(op["alpha"]) * torch.outer(u, coef)
            elif op["kind"] == "rope_shift":
                exp = model.paligemma_with_expert.gemma_expert.model
                inv = exp.rotary_emb.inv_freq.float().to(W.device)      # (head_dim / 2,)
                d2 = inv.numel()
                assert W.shape[0] % (2 * d2) == 0, (W.shape, d2)
                ang = float(op["delta"]) * inv
                c, s_ = ang.cos()[:, None], ang.sin()[:, None]
                W = W.clone()
                for h in range(W.shape[0] // (2 * d2)):                 # 每个头一块：前 d2 行 = x1，后 d2 行 = x2；x1' = c·x1 − s·x2，x2' = c·x2 + s·x1
                    a = h * 2 * d2
                    x1, x2 = W[a:a + d2].clone(), W[a + d2:a + 2 * d2].clone()
                    W[a:a + d2] = c * x1 - s_ * x2
                    W[a + d2:a + 2 * d2] = c * x2 + s_ * x1
            elif op["kind"] == "patch_blur":
                n = W.shape[-1]
                yy, xx = torch.meshgrid(torch.arange(n, device=W.device), torch.arange(n, device=W.device), indexing="ij")
                pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
                d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
                G = torch.exp(-d2 / (2 * float(op["sigma"]) ** 2))
                G = G / G.sum(1, keepdim=True)
                W = (W.reshape(W.shape[0], W.shape[1], -1) @ G).reshape(W.shape)
            elif op["kind"] == "patch_matrix":
                L = torch.as_tensor(z["L_%d" % i], device=W.device, dtype=torch.float32)
                if L.shape[0] == W.shape[-1] ** 2:
                    W = (W.reshape(W.shape[0], W.shape[1], -1) @ L).reshape(W.shape)
                else:
                    W = (W.reshape(W.shape[0], -1) @ L).reshape(W.shape)
                if "m_%d" % i in z.files:
                    bname = op["name"].replace(".weight", ".bias")
                    m = torch.as_tensor(z["m_%d" % i], device=W.device, dtype=torch.float32)
                    params[bname].data.copy_((params[bname].data.float() + before.reshape(before.shape[0], -1) @ m).to(params[bname].data.dtype))
            elif op["kind"] == "bias_add":
                W = W + torch.as_tensor(z["v_%d" % i], device=W.device, dtype=torch.float32)
            elif op["kind"] == "subspace_abl":
                U = torch.as_tensor(z["U_%d" % i], device=W.device, dtype=torch.float32)
                shape = W.shape
                M = W.reshape(shape[0], -1)
                W = (M - U @ (U.T @ M)).reshape(shape)
            elif op["kind"] == "input_abl":
                V = torch.as_tensor(z["V_%d" % i], device=W.device, dtype=torch.float32)
                W = W - float(op.get("alpha", 1.0)) * (W @ V) @ V.T
            elif op["kind"] == "scale":
                W = W * float(op["g"])
            elif op["kind"] == "out_row":
                W = W.clone()
                W[int(op["row"])] *= 1.0 + float(op["alpha"])
            else:
                raise ValueError(op["kind"])
            p.data.copy_(W.to(p.data.dtype))
            after = p.data.float()
            d = after - before
            rep = {"name": op["name"], "kind": op["kind"], "n_changed": int((d != 0).sum()), "n_total": int(d.numel()),
                   "rel_frob": float(d.norm() / before.norm()), "sha_before": _sha(before), "sha_after": _sha(after)}
            if op["kind"] == "ablate":                         # bfloat16 只有 8 位有效数字：看舍入后这个方向真的变了多少
                c0, c1 = u @ before, u @ after
                sel = coef != 0
                rep["proj_ratio_after"] = float((c1[sel] * c0[sel]).sum() / (c0[sel] ** 2).sum())
            report.append(rep)
    return report
