#!/usr/bin/env python
# ----------------------------------------------------------------------------
# G0.5 接入冒烟测试 —— 只验证不需要权重的部分
#
#   policy/g05/GalaxeaVLA/.venv/bin/python policy/g05/smoke_test.py
#
# 覆盖:
#   1. deploy_policy / g05_model 能 import（依赖装齐）
#   2. hydra 配置组装：sim_robotwin + task=robotwin -> 14 维 embodiment 切片
#   3. encode_obs 能把比赛格式的假观测转成 G0.5 输入
#   4. 动作 chunk 组装：部件 dict -> [T,14] 打平，切片位置正确
#   5. 比赛数据的 av1 视频能被解码（训练数据通路）
#
# 需要权重才能跑的部分（模型实例化、前向、真实推理）不在这里，会明确标 SKIP。
# ----------------------------------------------------------------------------

import os
import sys
import traceback
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = POLICY_DIR.parent.parent
sys.path.insert(0, str(POLICY_DIR))

PASS, FAIL, SKIP = [], [], []
CHECKS = []


def check(name):
    """只登记，不执行 —— 保证输出顺序是 标题 -> 逐条结果 -> 汇总。"""
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def run_checks(verbose_traceback: bool = False) -> None:
    for name, fn in CHECKS:
        try:
            result = fn()
            if result == "skip":
                SKIP.append(name)
                print(f"SKIP  {name}")
            else:
                PASS.append(name)
                print(f"ok    {name}" + (f"  ({result})" if isinstance(result, str) else ""))
        except Exception as exc:
            FAIL.append((name, exc))
            first = str(exc).splitlines()[0] if str(exc) else ""
            print(f"FAIL  {name}: {type(exc).__name__}: {first}")
            if verbose_traceback:
                traceback.print_exc()


@check("1. import deploy_policy / g05_model")
def _t1():
    import deploy_policy
    import g05_model
    for fn in ("get_model", "eval", "reset_model", "encode_obs"):
        assert hasattr(deploy_policy, fn), fn
    assert g05_model.CAMERA_KEYS == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    return "四个接口函数齐全"


@check("2. hydra 组装 sim_robotwin + task=robotwin")
def _t2():
    from g05_model import _compose_cfg
    cfg = _compose_cfg("sim_robotwin", "robotwin")

    emb = cfg.data.embodiment_datasets["robotwin"]
    assert str(emb.embodiment_type) == "robotwin"

    expected = [("left_arm", 0, 6), ("left_gripper", 6, 1),
                ("right_arm", 7, 6), ("right_gripper", 13, 1)]
    for group in ("action", "state"):
        got = [(str(m["key"]), int(m["start_index"]), int(m["raw_shape"]))
               for m in emb.shape_meta[group]]
        assert got == expected, f"{group} 切片不对: {got}"
        assert sum(d for _, _, d in got) == 14

    cams = [str(m["key"]) for m in emb.shape_meta["images"]]
    assert cams == ["cam_high", "cam_left_wrist", "cam_right_wrist"], cams
    for m in emb.shape_meta["images"]:
        assert list(m["raw_shape"]) == [3, 480, 640], m["raw_shape"]
    return f"14 维切片 + 3 路相机；action_size={int(cfg.data.action_size)}"


@check("3. encode_obs 处理比赛格式观测")
def _t3():
    import numpy as np
    from deploy_policy import encode_obs

    obs = {
        "sensor": {c: {"color": np.random.randint(0, 255, (1, 480, 640, 4), dtype=np.uint8)}
                   for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
        "robot": {"qpos": np.random.randn(1, 14).astype(np.float32)},
    }
    img_dict, state = encode_obs(obs)
    assert sorted(img_dict) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for k, v in img_dict.items():
        assert v.shape == (480, 640, 3) and v.dtype == np.uint8, (k, v.shape, v.dtype)
    assert state.shape == (14,), state.shape
    return "3x(480,640,3) uint8 + (14,)"


@check("4. 动作 chunk 组装 -> [T,14]")
def _t4():
    import numpy as np
    import torch
    from g05_model import G05

    action_meta = [
        {"key": "left_arm", "start_index": 0, "raw_shape": 6},
        {"key": "left_gripper", "start_index": 6, "raw_shape": 1},
        {"key": "right_arm", "start_index": 7, "raw_shape": 6},
        {"key": "right_gripper", "start_index": 13, "raw_shape": 1},
    ]
    fake = G05.__new__(G05)                     # 不跑 __init__，只测纯函数逻辑
    fake.action_meta = action_meta
    fake.action_dim = 14

    T = 5
    # 每个部件填上可识别的常数，方便验证落位
    pred = {
        "left_arm": torch.full((1, T, 6), 1.0),
        "left_gripper": torch.full((1, T, 1), 2.0),
        "right_arm": torch.full((1, T, 6), 3.0),
        "right_gripper": torch.full((1, T, 1), 4.0),
    }
    chunk = fake._assemble_chunk(pred)
    assert chunk.shape == (T, 14), chunk.shape
    row = chunk[0]
    assert np.allclose(row[0:6], 1.0), row
    assert np.allclose(row[6], 2.0), row
    assert np.allclose(row[7:13], 3.0), row
    assert np.allclose(row[13], 4.0), row

    # key 缺失必须报错，不能静默产出错位动作
    try:
        fake._assemble_chunk({k: v for k, v in pred.items() if k != "left_gripper"})
    except ValueError:
        pass
    else:
        raise AssertionError("缺 key 居然没报错")
    return "左臂0:6 左夹爪6 右臂7:13 右夹爪13 落位正确"


def _preload_npp() -> None:
    """把 nvidia-npp 的 .so 先 dlopen 进本进程。

    torchcodec 的原生库依赖 libnppicc.so.12，而 nvidia-npp-cu12 把它装在
    site-packages/nvidia/npp/lib 下——这个目录不在动态库搜索路径上。
    LD_LIBRARY_PATH 必须在进程启动前设好，本进程已经起来了改不了，
    但只要把这些 .so 先加载进来，后续 dlopen(libtorchcodec) 就能解析到依赖。
    eval.sh / finetune.sh 里走的是 LD_LIBRARY_PATH，这里是为了让冒烟测试能独立跑。
    """
    import ctypes
    import glob as _glob
    import sysconfig

    lib_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "npp" / "lib"
    for so in sorted(_glob.glob(str(lib_dir / "*.so*"))):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


@check("5. 比赛 av1 视频解码")
def _t5():
    ds_root = REPO_ROOT / "lerobot_dataset"
    videos = sorted(ds_root.glob("*/*/videos/observation.images.cam_high/chunk-000/*.mp4"))
    if not videos:
        return "skip"
    path = str(videos[0])
    _preload_npp()
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        return "skip"
    dec = VideoDecoder(path)
    frame = dec[0]
    assert frame.shape[0] == 3 and frame.shape[1] == 480 and frame.shape[2] == 640, frame.shape
    return f"torchcodec 解 av1 成功 {tuple(frame.shape)} <- {Path(path).parts[-5]}"


@check("6. 生成的 cobotmagic 配置可被 hydra 组装")
def _t6():
    from omegaconf import OmegaConf
    data_yaml = POLICY_DIR / "configs" / "data" / "cobotmagic.yaml"
    task_yaml = POLICY_DIR / "configs" / "task" / "cobotmagic.yaml"
    if not data_yaml.is_file():
        return "skip"
    cfg = OmegaConf.load(data_yaml)
    emb = cfg.embodiment_datasets["cobotmagic"]
    assert str(emb.lerobot_ds_version) == "3.0"
    expected = [("left_arm", 0, 6), ("left_gripper", 6, 1),
                ("right_arm", 7, 6), ("right_gripper", 13, 1)]
    got = [(str(m["key"]), int(m["start_index"]), int(m["raw_shape"]))
           for m in emb.shape_meta["action"]]
    assert got == expected, got
    dirs = list(emb.dataset_groups[0].dataset_dirs)
    for d in dirs:
        assert (REPO_ROOT / d / "meta" / "info.json").is_file(), d
    task = OmegaConf.load(task_yaml)
    assert int(task.model.processor.num_output_cameras) == 3
    return f"{len(dirs)} 个数据集目录全部存在"


@check("7. 走评估脚本真实加载路径（含仿真栈）")
def _t7():
    """eval_policy.load_policy_adapter("g05") 是比赛真正用的加载方式。

    它把 RoboSynChallenge/policy 加进 sys.path，导致裸 `import g05` 会命中
    本适配器目录而不是 GalaxeaVLA 的 src/g05 —— 曾经因此循环导入炸掉。
    前 6 项测试用的是另一种导入方式，测不出这个，所以单独加这一项。
    """
    import os

    embodichain = Path(os.environ.get("EMBODICHAIN_ROOT", "/home/phl/workspace/EmbodiChain"))
    for p in (str(REPO_ROOT), str(embodichain)):
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        import eval_policy
    except ImportError:
        return "skip"   # 仿真栈没装齐

    pkg = eval_policy.load_policy_adapter("g05")
    for fn in ("get_model", "eval", "reset_model"):
        assert hasattr(pkg, fn), fn

    import g05 as galaxea
    assert str(Path(galaxea.__file__).resolve()).startswith(
        str((POLICY_DIR / "GalaxeaVLA" / "src").resolve())
    ), f"顶层 g05 被适配器目录劫持了: {galaxea.__file__}"
    return f"{pkg.__name__} 三函数齐全；g05 正确指向 GalaxeaVLA/src"


def main() -> int:
    verbose = "-v" in sys.argv
    print("=" * 60)
    print("G0.5 接入冒烟测试（不需要权重）")
    print("=" * 60)
    run_checks(verbose_traceback=verbose)
    print("=" * 60)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}  SKIP={len(SKIP)}")
    if SKIP:
        print(f"跳过: {', '.join(SKIP)}")
    if FAIL:
        print("失败:")
        for name, exc in FAIL:
            first = str(exc).splitlines()[0] if str(exc) else ""
            print(f"  - {name}: {type(exc).__name__}: {first}")
        print("（加 -v 看完整堆栈）")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
