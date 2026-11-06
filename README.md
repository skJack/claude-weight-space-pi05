# Claude Programs a Robot in Weight Space · Claude 在权重空间里给机器人编程

**Read the post: [English](https://skjack.github.io/claude-weight-space-pi05/) · [中文](https://skjack.github.io/claude-weight-space-pi05/zh.html)**

No gradients and no training. Claude Opus 5.5 changed one matrix inside π0.5, the projection layer between SigLIP and the backbone, and success on 50 RoboCasa tasks rose from 51.9% to 65.1% (1,000 episode pairs on fresh seeds, +219 / −87). The idea follows MIT's [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (a blog post, not peer reviewed).

Simulation only, not peer reviewed. These are 50 short tasks, not the official `target50`. Claude found the fix by comparing training and evaluation images, not by reading the weights, and re-encoding frames with x264 in the loop works about as well without touching any weights. The post has the full caveats.

不用梯度，不训练。Claude Opus 5.5 改了 π0.5 里的一个矩阵，也就是 SigLIP 和骨干之间的投影层，50 个 RoboCasa 任务的成功率从 51.9% 升到 65.1%（1000 对新种子，+219 / −87）。这个想法来自 MIT 的 [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)（博客，未经同行评审）。

只在仿真里测过，未经同行评审。这是 50 个短任务，不是官方的 `target50`。Claude 是通过对比训练和评测的图像找到修法的，不是靠读权重；不改权重、在闭环里把画面过一遍 x264，效果也差不多。完整的局限见博客。

## Apply the edit

```python
# model = PyTorch pi0.5, converted from pi05_pretrain_human300 with the openpi script
# (robocasa/robocasa365_checkpoints on Hugging Face, multitask_learning/75000)
from apply_edit import apply_edit   # edit/apply_edit.py
print(apply_edit(model, "edit/edit.npz"))
```

The report should show `rel_frob` 0.0808, and SHA-256 prefixes `c3c6ae81d4fa4956` before and `102128668f8b1ba5` after.

## What's here

| Path | Contents |
|---|---|
| `index.html`, `zh.html`, `assets/` | The post (GitHub Pages serves the repository root) |
| `edit/` | The edit, the script that applies it, and its manifest |
| `results/` | Per-episode outcomes on fresh seeds and the test set, every practice run, the official-stack check, offline controls |
| `code/` | The scripts for the main steps. See `code/README.md` |
| `site_src/` | Sources of the two pages. `python3 site_src/build.py` rebuilds them from `results/` |

## Citation · 引用

If you'd like to cite this post, please use: · 如果要引用这篇文章，请使用以下格式：

Jack Sun Kelip. "Claude Programs a Robot in Weight Space." Blog post, November 2026. https://skjack.github.io/claude-weight-space-pi05/

```bibtex
@misc{sun2026claude,
  author       = {{Jack Sun Kelip}},
  title        = {Claude Programs a Robot in Weight Space},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://skjack.github.io/claude-weight-space-pi05/}},
  note         = {Blog post}
}
```
