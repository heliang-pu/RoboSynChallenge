#!/usr/bin/env python3
"""Write per-episode ``episode_success`` labels into a LeRobot v3.0 dataset.

sim-RECAP 数据桥接(Phase B):把 eval_policy.py --rollout_save 产出的
episode_success.json 边车标签,写入数据集 meta/episodes 下的 parquet,
成为 Evo-RL 价值函数训练所要求的 ``episode_success`` 列
(字符串 "success"/"failure",按 episode_index 对齐)。

用法:
  # rollout 数据集:按边车文件逐集打标(默认读 <dataset>/episode_success.json)
  python scripts/label_rollout_dataset.py --dataset lerobot_dataset/rollouts/<name>

  # 专家数据集:整体打成 success(合并进数据池前必须有显式标签)
  python scripts/label_rollout_dataset.py --dataset lerobot_dataset/click_bell --constant success

校验门(任何一条不过即非零退出,不写任何文件):
  * 数据集必须是 LeRobot v3.0(meta/info.json 的 codebase_version)
  * 边车模式下:边车 episode 数 == 数据集 total_episodes ==
    边车记录的 saved_episode_count(若存在)
  * episode_index 连续且与边车一一对应

依赖仅 pyarrow(仿真环境自带);幂等,重复运行会覆盖已有标签列。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SUCCESS = "success"
FAILURE = "failure"
COLUMN = "episode_success"


def fail(msg: str) -> None:
    print(f"[label_rollout_dataset] 错误: {msg}", file=sys.stderr)
    sys.exit(1)


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        fail(f"不是 LeRobot 数据集(缺 {info_path})")
    with open(info_path) as f:
        info = json.load(f)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        fail(
            f"codebase_version={version},只支持 v3.0;"
            "v2.1 数据请先用 LeRobot 3.0 重新转换或在转换前打标"
        )
    return info


def read_sidecar_records(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("episodes", [])


def load_labels_from_sidecar(
    sidecar_path: Path,
    total_episodes: int,
    prefix_success: int = 0,
    prefix_labels: dict[int, str] | None = None,
) -> dict[int, str]:
    if not sidecar_path.exists():
        fail(
            f"找不到边车标签 {sidecar_path};"
            "rollout 采集请用 eval_policy.py 的 --rollout_save True,"
            "专家数据请改用 --constant success"
        )
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    episodes = sidecar.get("episodes", [])
    saved_count = sidecar.get("saved_episode_count")
    expected_from_sidecar = total_episodes - prefix_success

    # 三方一致性:边车条数、数据集总集数(扣掉专家前缀)、采集端记录的落盘数
    if len(episodes) != expected_from_sidecar:
        fail(
            f"边车有 {len(episodes)} 条标签,但数据集扣除 {prefix_success} 个专家前缀后"
            f"还有 {expected_from_sidecar} 个 episode;"
            "采集与落盘不对齐,该数据集不可用于价值训练"
        )
    if saved_count is not None and int(saved_count) != len(episodes):
        fail(
            f"采集端记录 saved_episode_count={saved_count},"
            f"与边车标签条数 {len(episodes)} 不一致"
        )

    # 合并数据集布局: [0, prefix_success) 是前缀数据集的 episode,
    # 之后依次是 rollout episode(边车 index + 偏移)。
    # 前缀标签: 纯专家数据全 success;若前缀本身是上一轮的合并池
    # (混有失败集),由 prefix_labels 显式给出。
    if prefix_labels is None:
        labels: dict[int, str] = {i: SUCCESS for i in range(prefix_success)}
    else:
        if sorted(prefix_labels) != list(range(prefix_success)):
            fail(
                f"前缀边车覆盖 {len(prefix_labels)} 个 episode,"
                f"与前缀数据集的 {prefix_success} 个不一致"
            )
        labels = dict(prefix_labels)
    for rec in episodes:
        idx = int(rec["episode_index"]) + prefix_success
        if idx in labels and idx >= prefix_success:
            fail(f"边车里 episode_index={rec['episode_index']} 重复")
        labels[idx] = SUCCESS if rec["success"] else FAILURE
    if sorted(labels) != list(range(total_episodes)):
        fail("边车 episode_index 不连续,无法与数据集对齐")
    return labels


def write_labels(dataset_root: Path, labels: dict[int, str] | str) -> None:
    parquet_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not parquet_files:
        fail(f"{dataset_root}/meta/episodes 下没有 parquet 文件")

    n_success = 0
    n_total = 0
    for path in parquet_files:
        table = pq.read_table(path)
        episode_indices = table.column("episode_index").to_pylist()
        if isinstance(labels, str):
            values = [labels] * len(episode_indices)
        else:
            missing = [i for i in episode_indices if i not in labels]
            if missing:
                fail(f"{path} 中的 episode_index {missing[:5]}... 没有对应标签")
            values = [labels[i] for i in episode_indices]

        n_success += sum(v == SUCCESS for v in values)
        n_total += len(values)

        if COLUMN in table.column_names:
            table = table.drop_columns([COLUMN])
        table = table.append_column(COLUMN, pa.array(values, type=pa.string()))
        pq.write_table(table, path, compression="snappy")

    print(
        f"[label_rollout_dataset] 完成: {dataset_root}\n"
        f"  写入列: {COLUMN}(meta/episodes 共 {len(parquet_files)} 个 parquet)\n"
        f"  标签分布: success={n_success}  failure={n_total - n_success}  "
        f"({100 * n_success / max(n_total, 1):.1f}% success)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="LeRobot v3.0 数据集根目录")
    parser.add_argument(
        "--sidecar",
        default=None,
        help="episode_success.json 路径,默认 <dataset>/episode_success.json",
    )
    parser.add_argument(
        "--constant",
        choices=[SUCCESS, FAILURE],
        default=None,
        help="不读边车,把全部 episode 打成同一标签(用于专家数据集)",
    )
    parser.add_argument(
        "--prefix-success",
        type=int,
        default=0,
        help="合并数据集用:前 N 个 episode 是专家数据(全 success),"
        "其余按边车标签对齐(边车 index + N)",
    )
    parser.add_argument(
        "--prefix-sidecar",
        default=None,
        help="前缀部分不是纯专家而是上一轮合并池时,给出它的 episode_success.json,"
        "前缀标签按此文件而非全 success",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    info = load_info(dataset_root)
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        fail("数据集 total_episodes 为 0")

    if args.constant is not None:
        write_labels(dataset_root, args.constant)
        return

    sidecar_path = (
        Path(args.sidecar).expanduser().resolve()
        if args.sidecar
        else dataset_root / "episode_success.json"
    )
    prefix_labels = None
    if args.prefix_sidecar:
        prefix_labels = {
            int(rec["episode_index"]): (SUCCESS if rec["success"] else FAILURE)
            for rec in read_sidecar_records(Path(args.prefix_sidecar))
        }

    labels = load_labels_from_sidecar(
        sidecar_path,
        total_episodes,
        prefix_success=max(0, args.prefix_success),
        prefix_labels=prefix_labels,
    )
    write_labels(dataset_root, labels)


if __name__ == "__main__":
    main()
