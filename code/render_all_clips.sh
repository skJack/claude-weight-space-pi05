#!/bin/bash
# 9 组并排视频，规则固定：只用初始状态逐位相同的配对，每个任务取局号最小的一局。上升的取"base 失败、post_k3 成功"，变差的取"base 成功、post_k3 失败"，都失败的取都失败。
R=/path/to/robocasa_root; O=/path/to/work/blog/out/video; mkdir -p "$O"; L=/path/to/work/logs/blog_render.log; : > "$L"
JOBS=(
"PickPlaceCounterToBlender others 0 blender"
"CoffeeServeMug rep 0 coffee"
"PickPlaceMicrowaveToCounter rep 1 microwave"
"TurnOnElectricKettle rep 5 kettle"
"NavigateKitchen others 4 navigate"
"CloseToasterOvenDoor rep 1 toaster"
"TurnOnBlender rep 0 worse_blender"
"PickPlaceCounterToToasterOven others 0 worse_toaster"
"OpenOven rep 0 both_oven"
)
DEV=(0 1)   # EGL device ids; depends on the machine
i=0; PIDS=()
for j in "${JOBS[@]}"; do
  set -- $j
  cd /tmp || exit 1
  __GL_SHADER_DISK_CACHE=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${DEV[$((i % 2))]} setsid nohup $R/envs/sim/bin/python -O code/render_clip.py "$1" "$2" "$3" "$O" "$4" >> "$L" 2>&1 < /dev/null &
  PIDS+=($!)
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=(); fi
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo "BLOG RENDER DONE $(date)" >> "$L"
