"""启动 RoboCasa π0.5 服务端（robocasa-benchmark/openpi@ca4c6d7 的 scripts/serve_policy.py），不改 fork 的代码。

问题：fork 的 LeRobotRobocasaDataConfig.create() 在找不到 asset_id 下的 norm stats 时，会去训练数据目录
（robocasa/datasets/v1.0/pretrain/.../lerobot/meta/stats.json）合并统计量；我们没下训练数据，直接 FileNotFoundError。
而 policy_config.create_trained_policy 在 asset_id 为 None 时本来就从 checkpoint/assets/norm_stats.json 读（作者改过）。
所以这里把那两个"从数据目录读统计量"的函数换成返回 None，让它走 checkpoint 自带的统计量。
另外 robocasa/__init__.py 断言 numpy==2.2.5，而 openpi 环境锁的是 1.26.4，服务端只在 import 时碰到它 → 用 python -O 跑。

用法：
  cd <R>/openpi && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false <R>/envs/openpi/bin/python -O <本文件> \
      --port 8021 policy:checkpoint --policy.config=pi05_pretrain_human300 --policy.dir=<R>/ckpt/pi05_pretrain_human300/multitask_learning/75000
"""
import logging
import os
import sys

import tyro

import openpi.groot_utils.groot_openpi_dataset as _g

_g._load_norm_stats_from_groot_mixture_dataset = lambda *a, **k: None
_g._load_norm_stats_from_groot_dataset = lambda *a, **k: None

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import serve_policy  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    serve_policy.main(tyro.cli(serve_policy.Args))
