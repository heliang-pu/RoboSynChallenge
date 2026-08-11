"""从 wandb offline datastore 里读训练 loss 曲线（离线跑训练时看进度用）。

用法: python read_train_loss.py [run 目录]
不传参数就自动取最新的 offline-run-*。
"""

import glob
import json
import os
import sys

DEFAULT_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Xiaomi-Robotics-1", "xr1", "wandb", "offline-run-*",
)

run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(DEFAULT_GLOB))[-1]
wandb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
if not wandb_files:
    raise SystemExit(f"{run_dir} 下没有 .wandb 文件")

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

store = datastore.DataStore()
store.open_for_scan(wandb_files[0])

records = []
while True:
    try:
        data = store.scan_data()
    except Exception:
        break
    if data is None:
        break
    record = wandb_internal_pb2.Record()
    record.ParseFromString(data)
    if record.WhichOneof("record_type") != "history":
        continue
    row = {}
    for item in record.history.item:
        key = item.key or ".".join(item.nested_key)
        try:
            row[key] = json.loads(item.value_json)
        except Exception:
            row[key] = item.value_json
    if row:
        records.append(row)

if not records:
    raise SystemExit("还没有 history 记录（wandb 会缓冲一会儿再落盘）")

columns = ["trainer/global_step", "train/loss", "train/loss_mse", "train/loss_freq",
           "train/loss_l1", "train/loss_score", "lr", "train/token"]
print(f"run: {os.path.basename(run_dir)}   记录数: {len(records)}")
print(f"{'step':>7} {'loss':>9} {'mse':>9} {'freq':>9} {'l1':>9} {'score':>9} {'lr':>10} {'token':>7}")
for row in records:
    values = []
    for key in columns:
        value = row.get(key)
        if value is None:
            values.append("-")
        elif key == "trainer/global_step" or key == "train/token":
            values.append(f"{int(value)}")
        elif key == "lr":
            values.append(f"{value:.3e}")
        else:
            values.append(f"{value:.5f}")
    print(f"{values[0]:>7} {values[1]:>9} {values[2]:>9} {values[3]:>9} "
          f"{values[4]:>9} {values[5]:>9} {values[6]:>10} {values[7]:>7}")

if len(records) >= 2:
    first = records[0].get("train/loss")
    last = records[-1].get("train/loss")
    if first is not None and last is not None:
        trend = "下降" if last < first else ("上升" if last > first else "持平")
        print(f"\ntrain/loss: {first:.5f} -> {last:.5f}  ({trend})")
