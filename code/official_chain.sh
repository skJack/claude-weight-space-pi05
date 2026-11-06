#!/bin/bash
# 官方评测栈核对 base：官方 JAX 权重 + robocasa-benchmark/openpi 的 serve_policy（经 serve_rc.py：只是让它读 checkpoint 自带的 norm stats、用 -O 绕开 numpy 版本断言，不改推理）
# + 官方客户端 eval_env，split=target，时长 ×1.5，每 5 步重规划。
R=/path/to/robocasa_root; W=/path/to/work; O=$W/official; rm -rf $O; mkdir -p $O
CK=$R/ckpt/pi05_pretrain_human300/multitask_learning/75000
EGL=(0 1 2 3 4 5 6 7)   # EGL device id for each GPU; depends on the machine
TASKS=(CloseToasterOvenDoor TurnOnElectricKettle CoffeeServeMug PickPlaceCounterToBlender OpenCabinet CloseMicrowave PickPlaceMicrowaveToCounter TurnOnMicrowave)
GP=(0 1 2 3)
SRV=""
cd $R/openpi || exit 1
ls code/serve_rc.py || exit 1
for g in "${GP[@]}"; do
  CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=false setsid nohup $R/envs/openpi/bin/python -O code/serve_rc.py --port $((8100+g)) policy:checkpoint --policy.config=pi05_pretrain_human300 --policy.dir=$CK > $O/server_$g.log 2>&1 < /dev/null &
  SRV="$SRV $!"
done
for g in "${GP[@]}"; do
  for i in $(seq 1 120); do ss -ltn | grep -q ":$((8100+g)) " && break; sleep 5; done
  ss -ltn | grep -q ":$((8100+g)) " || { echo "server $g 没起来"; tail -n 5 $O/server_$g.log; kill $SRV; exit 1; }
done
echo "$(date) servers up"
cd /tmp || exit 1
CLI=""
for i in "${!TASKS[@]}"; do
  g=${GP[$((i % 4))]}
  __GL_SHADER_DISK_CACHE=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${EGL[$g]} setsid nohup $R/envs/sim/bin/python -O $W/code/official_client.py ${TASKS[$i]} $((8100+g)) 20 $O/logs > $O/client_${TASKS[$i]}.log 2>&1 < /dev/null &
  CLI="$CLI $!"
done
for x in $CLI; do while kill -0 $x 2>/dev/null; do sleep 20; done; done
kill $SRV
echo "$(date) OFFICIAL CHAIN DONE"
