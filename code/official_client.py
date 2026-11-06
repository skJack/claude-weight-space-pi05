"""用官方 examples/robocasa/main.py 的 eval_env 原样跑（官方客户端 + 官方 JAX 服务端），只是绕开它 eval_main 里读 args.task_soup 的那行。
用法：python official_client.py <task> <port> <n> <log_dir>"""
import sys, runpy, os, types
sys.modules.setdefault("tyro", types.ModuleType("tyro"))     # tyro 只用于官方脚本 __main__ 里解析命令行，这里不走那条路
sys.path.insert(0, "/path/to/robocasa_root/openpi/examples/robocasa")
task, port, n, log_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
g = runpy.run_path("/path/to/robocasa_root/openpi/examples/robocasa/main.py", run_name="not_main")
g["eval_env"](task, "target", log_dir, n, 224, 5, "127.0.0.1", port, 7)
print("OFFICIAL DONE", task, flush=True)
