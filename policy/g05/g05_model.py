# ----------------------------------------------------------------------------
# G0.5 (GalaxeaVLA) 进程内推理封装
#
# 官方部署方式是 scripts/serve_policy.py 起 WebSocket 服务，客户端发 msgpack。
# 比赛评估脚本要求 policy 在同一进程里跑，所以这里绕开 WebSocket，
# 直接复用官方的 g05.models.g05.inferencer.PolicyInferencer。
#
# 调用链（与官方 experiments/robotwin/galaxeafm_policy/deploy_policy.py 一致）:
#   ckpt/.hydra/config.yaml ──> cfg.model / cfg.tokenizer
#   hydra compose(sim_robotwin, task=robotwin) ──> cfg.data / cfg.EVALUATION
#   instantiate(cfg.model.model_arch) + load_state_dict_safely(ckpt)
#   build_processors(cfg) ──> MixtureProcessor{"robotwin": GalaxeaCoTProcessor}
#   PolicyInferencer(policy, processor).infer([obs_dict]) ──> {part_key: [T, D]}
#
# 维度约定见 README_INTEGRATION.md：robotwin embodiment 原生就是 14 维，
# 与比赛的 (1,14) 绝对关节位置逐位对应，不需要 27→14 的重映射。
# ----------------------------------------------------------------------------

import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# GalaxeaVLA 源码根目录：policy/g05/GalaxeaVLA
G05_ROOT = Path(__file__).resolve().parent / "GalaxeaVLA"
G05_SRC = G05_ROOT / "src"
for _p in (str(G05_ROOT), str(G05_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _bind_galaxea_g05_package() -> None:
    """把 GalaxeaVLA 的顶层包 `g05` 按文件路径显式绑定到 sys.modules["g05"]。

    名字冲突：我们这个适配器目录也叫 `policy/g05/`，而 scripts/eval_policy.py 会把
    `RoboSynChallenge/policy` 加进 sys.path。于是裸写 `import g05` 会命中**我们自己**
    的适配器包，而不是 GalaxeaVLA 的 `src/g05`，进而触发循环导入：

        policy/g05/__init__.py -> deploy_policy -> g05_model
          -> `from g05.models...` 命中 policy/g05/__init__.py（还没初始化完）
          -> ImportError: partially initialized module 'g05_model'

    只靠调 sys.path 顺序不保险（谁先被加进去取决于调用方），这里直接按绝对路径把
    真正的 GalaxeaVLA 包注册进 sys.modules，让后续所有 `from g05.xxx import` 都确定
    落到 GalaxeaVLA 上。
    """
    import importlib.util

    existing = sys.modules.get("g05")
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if existing_file.startswith(str(G05_SRC)):
            return  # 已经是 GalaxeaVLA 那个了

    pkg_dir = G05_SRC / "g05"
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise ImportError(
            f"找不到 GalaxeaVLA 的 g05 包: {init_py}\n"
            "请确认已 git clone OpenGalaxea/GalaxeaVLA 到 policy/g05/GalaxeaVLA"
        )

    spec = importlib.util.spec_from_file_location(
        "g05", init_py, submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["g05"] = module
    spec.loader.exec_module(module)


_bind_galaxea_g05_package()

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g05.data_processor.processor.mixture_processor import MixtureProcessor
from g05.data_processor.transforms.action_filter import BaseActionFilter
from g05.models.g05.inferencer import PolicyInferencer
from g05.utils.checkpoint.checkpoint_utils import load_state_dict_safely
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.utils.data.processor_utils import build_processors

logger = logging.getLogger(__name__)

# 比赛环境的三路相机 -> G0.5 robotwin embodiment 的图像 key。
# 两边名字本来就一样，这里显式列出来是为了让 key 不匹配时立刻报错而不是静默少喂一路图。
CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

_MIXED_PRECISION_DTYPE = {
    "no": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _resolve_under_root(raw: Any) -> Optional[Path]:
    """把配置里的相对路径按 GalaxeaVLA 根目录解析成绝对路径；null/空返回 None。"""
    if _is_none_like(raw):
        return None
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (G05_ROOT / path)


def _meta_dim(meta: Dict[str, Any]) -> int:
    """state/action 的 raw_shape 是标量维度（图像的才是 [C,H,W] 列表）。"""
    raw_shape = meta["raw_shape"]
    if isinstance(raw_shape, int):
        return int(raw_shape)
    raise ValueError(f"state/action meta 的 raw_shape 必须是整数，实际是: {raw_shape}")


def _flat_dim_from_meta(meta_list: List[Dict[str, Any]]) -> int:
    """按 start_index + raw_shape 校验切片无重叠无空洞，返回打平后的总维度。"""
    max_end = 0
    for meta in meta_list:
        max_end = max(max_end, int(meta["start_index"]) + _meta_dim(meta))

    occupied = np.zeros(max_end, dtype=bool)
    for meta in meta_list:
        start = int(meta["start_index"])
        end = start + _meta_dim(meta)
        if occupied[start:end].any():
            raise ValueError(f"shape_meta 切片重叠: {meta}")
        occupied[start:end] = True
    if not occupied.all():
        raise ValueError(f"shape_meta 切片未覆盖下标: {np.where(~occupied)[0].tolist()}")
    return int(max_end)


def _split_flat_vector(
    vector: np.ndarray, meta_list: List[Dict[str, Any]], *, repeat_steps: int
) -> Dict[str, torch.Tensor]:
    """14 维打平 qpos -> {left_arm: [T,6], left_gripper: [T,1], ...}"""
    payload: Dict[str, torch.Tensor] = {}
    for meta in meta_list:
        start = int(meta["start_index"])
        dim = _meta_dim(meta)
        part = vector[start : start + dim]
        payload[str(meta["key"])] = (
            torch.from_numpy(np.ascontiguousarray(part))
            .unsqueeze(0)
            .expand(repeat_steps, -1)
            .float()
        )
    return payload


def _zero_action_by_meta(meta_list: List[Dict[str, Any]], *, horizon: int) -> Dict[str, torch.Tensor]:
    """推理时 action 是占位（processor 要求这个 key 存在），全零 + action_is_pad 全 True。"""
    return {
        str(meta["key"]): torch.zeros(horizon, _meta_dim(meta), dtype=torch.float32)
        for meta in meta_list
    }


def _as_chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"图像必须是 HWC 三通道，实际 shape={image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


# ----------------------------------------------------------------------------
# Hydra 配置组装
# ----------------------------------------------------------------------------

def _register_hydra_builtin_resolvers() -> None:
    """脱离 hydra.main 手工 compose 时，这些内置 resolver 要自己注册。"""
    OmegaConf.register_new_resolver(
        "now", lambda pattern, _tz="": time.strftime(pattern), replace=True
    )
    OmegaConf.register_new_resolver(
        "oc.env",
        lambda key, default=None: (
            os.environ[key]
            if key in os.environ
            else default
            if default is not None
            else (_ for _ in ()).throw(KeyError(f"环境变量 '{key}' 未设置"))
        ),
        replace=True,
    )


def _register_project_oc_load_resolver() -> None:
    """configs 里的 ${oc.load:configs/...} 是相对 GalaxeaVLA 根目录的，这里固定到 G05_ROOT。"""

    def _oc_load_from_project(path: str, key: Optional[str] = None) -> Any:
        load_path = Path(path)
        if not load_path.is_absolute():
            load_path = (G05_ROOT / load_path).resolve()
        cfg = OmegaConf.load(load_path)
        if key is None or key == "":
            return cfg
        return OmegaConf.select(cfg, key)

    OmegaConf.register_new_resolver("oc.load", _oc_load_from_project, replace=True)


def _compose_cfg(sim_cfg_name: str, sim_task: str) -> DictConfig:
    """组装 cfg.data / cfg.EVALUATION。

    直接复用官方 configs/sim_robotwin.yaml（defaults: train），用 task=robotwin 取到
    14 维双臂 embodiment 定义。不做全局 resolve —— sim 配置里有 ${hydra:...} 字段，
    本进程不是 hydra 启动的，解析会炸；下面只读不依赖 hydra runtime 的字段。
    """
    register_default_resolvers()
    _register_hydra_builtin_resolvers()
    _register_project_oc_load_resolver()

    configs_root = (G05_ROOT / "configs").resolve()
    config_name = sim_cfg_name[:-5] if sim_cfg_name.endswith(".yaml") else sim_cfg_name
    overrides = [] if _is_none_like(sim_task) else [f"task={sim_task}"]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    prev_cwd = Path.cwd()
    os.chdir(G05_ROOT)
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(config_name=config_name, overrides=overrides)
    finally:
        os.chdir(prev_cwd)
    return cfg


def _apply_checkpoint_model_config(cfg: DictConfig, checkpoint_file: Path) -> DictConfig:
    """模型结构和 action tokenizer 配置以 checkpoint 自带的 .hydra/config.yaml 为准。

    布局: <run_dir>/checkpoints/model_state_dict.pt + <run_dir>/.hydra/config.yaml
    """
    run_dir = checkpoint_file.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"找不到 checkpoint 的 hydra 配置: {config_path}\n"
            f"期望布局: {run_dir}/checkpoints/model_state_dict.pt 与 {run_dir}/.hydra/config.yaml"
        )

    ckpt_cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(ckpt_cfg, False)
    for key in ("model", "tokenizer"):
        if ckpt_cfg.get(key) is None:
            raise KeyError(f"checkpoint 配置缺少 `{key}`: {config_path}")

    OmegaConf.set_struct(cfg, False)
    cfg.model = OmegaConf.create(OmegaConf.to_container(ckpt_cfg.model, resolve=False))
    cfg.tokenizer = OmegaConf.create(OmegaConf.to_container(ckpt_cfg.tokenizer, resolve=False))

    # 早期内部版本的 target 前缀是 galaxea_fm.*，开源版已改名 g05.*，混用会在
    # instantiate 时才炸且报错难懂，这里提前拦。
    blob = str(OmegaConf.to_container(cfg.model, resolve=False)) + str(
        OmegaConf.to_container(cfg.tokenizer, resolve=False)
    )
    if "galaxea_fm." in blob:
        raise ValueError(f"checkpoint 配置还在引用旧的 `galaxea_fm.*`，需要先修 bundle: {config_path}")

    # 三个外部资源路径：HF processor / processor tokenizer / action tokenizer
    hf_processor_path = _resolve_under_root(cfg.model.model_arch.hf_processor_path)
    if hf_processor_path is None or not (hf_processor_path / "tokenizer.json").is_file():
        raise FileNotFoundError(f"hf_processor 的 tokenizer.json 不存在: {hf_processor_path}")

    # checkpoint 落盘的配置里，processor 的 tokenizer 路径可能是 null。
    # 仓库的 configs/model/g05.yaml 里这个字段本来是
    #   pretrained_model_name_or_path: ${model.model_arch.hf_processor_path}
    # 的插值，训练时被 resolve 掉、存成 null（官方 g05-base 就是 null）。
    # 这里按原意回填成 model_arch 那个路径，而不是拿 "None" 去拼路径。
    tok_key = "model.processor.tokenizer_params.pretrained_model_name_or_path"
    tok_raw = OmegaConf.select(cfg, tok_key)
    if _is_none_like(tok_raw):
        OmegaConf.update(cfg, tok_key, str(hf_processor_path), force_add=True)
        logger.info("checkpoint 里 processor tokenizer 路径为空，按原插值回填为 %s", hf_processor_path)
    else:
        tok_path = _resolve_under_root(tok_raw)
        if tok_path is None or not (tok_path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"processor.tokenizer 的 tokenizer.json 不存在: {tok_path}")

    at_path = _resolve_under_root(cfg.tokenizer.vq_config.ckpt_dir)
    if at_path is None or not at_path.is_file():
        raise FileNotFoundError(f"action tokenizer 权重不存在: {at_path}")

    if OmegaConf.select(cfg, "model.tokenizer.vq_config") is None:
        raise KeyError("checkpoint 配置缺少 `model.tokenizer.vq_config`")

    logger.info("模型配置来自 checkpoint: %s", config_path)
    return cfg


def _resolve_dataset_stats_path(explicit: Optional[str], checkpoint_file: Path) -> Path:
    """显式指定优先；否则从 checkpoint 逐级往上找 dataset_stats.json（官方约定）。"""
    if not _is_none_like(explicit):
        path = Path(str(explicit)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset_stats 不存在: {path}")
        return path

    for parent in list(checkpoint_file.parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            logger.info("自动找到 dataset_stats: %s", candidate)
            return candidate
    raise FileNotFoundError(
        f"未能从 {checkpoint_file} 的上级目录自动找到 dataset_stats.json，"
        "请在 deploy_policy.yml 里显式配置 dataset_stats_path"
    )


# ----------------------------------------------------------------------------
# 推理封装
# ----------------------------------------------------------------------------

class G05:
    """G0.5 策略封装。

    对外四个方法（与 policy/pi05/pi_model.py 的 PI0 保持一致的用法）:
        set_language(instruction)             设置语言指令
        update_observation_window(imgs, state) 塞入当前观测
        get_action()                          返回 [T, 14] 动作 chunk
        reset_obsrvationwindows()             episode 之间清状态
    """

    def __init__(
        self,
        ckpt_path: str,
        dataset_stats_path: Optional[str] = None,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        action_horizon: Optional[int] = None,
        replan_steps: int = 16,
        num_inference_steps: Optional[int] = None,
        control_frequency: float = 25.0,
        sim_cfg_name: str = "sim_robotwin",
        sim_task: str = "robotwin",
        embodiment: str = "robotwin",
        seed: Optional[int] = None,
    ) -> None:
        checkpoint_file = Path(str(ckpt_path)).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"checkpoint 不存在: {checkpoint_file}\n"
                "OpenGalaxea/G05 是 gated 仓库，需要先在 HF 网页同意协议再下载，"
                "详见 policy/g05/README_INTEGRATION.md"
            )

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        precision = str(mixed_precision).strip().lower()
        if precision not in _MIXED_PRECISION_DTYPE:
            raise ValueError(
                f"mixed_precision 只能是 {sorted(_MIXED_PRECISION_DTYPE)}，实际是 {mixed_precision}"
            )
        model_dtype = _MIXED_PRECISION_DTYPE[precision]

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 cpu")
            device = "cpu"
        self.device = str(device)
        self.embodiment = str(embodiment)

        cfg = _compose_cfg(sim_cfg_name=sim_cfg_name, sim_task=sim_task)
        cfg = _apply_checkpoint_model_config(cfg, checkpoint_file)
        stats_path = _resolve_dataset_stats_path(dataset_stats_path, checkpoint_file)

        # 模型配置里的 hf_processor_path / action tokenizer 路径都是相对 GalaxeaVLA 根目录的，
        # 构造期间必须把 cwd 钉在那里。
        prev_cwd = Path.cwd()
        os.chdir(G05_ROOT)
        try:
            self._build(cfg, checkpoint_file, stats_path, model_dtype)
        finally:
            os.chdir(prev_cwd)

        # ---- 推理节奏 ----
        model_horizon = int(cfg.data.action_size)
        self.model_action_horizon = model_horizon
        horizon = model_horizon if action_horizon is None else int(action_horizon)
        if not 0 < horizon <= model_horizon:
            raise ValueError(f"action_horizon 必须在 (0, {model_horizon}]，实际是 {horizon}")
        self.exec_action_horizon = horizon
        self.replan_steps = int(max(1, min(int(replan_steps), horizon)))

        if num_inference_steps is not None:
            fm_helper = getattr(self.policy.model, "fm_helper", None)
            if fm_helper is None:
                raise ValueError("当前模型没有 fm_helper，无法设置 num_inference_steps")
            fm_helper.num_inference_steps = int(num_inference_steps)

        self.control_frequency = float(control_frequency)
        self.instruction: Optional[str] = None
        self.observation_window: Optional[Dict[str, Any]] = None
        self._logged_image_shapes: set = set()

        logger.info(
            "G0.5 就绪 | ckpt=%s | stats=%s | horizon=%d | replan=%d | freq=%.1f | device=%s",
            checkpoint_file, stats_path, self.exec_action_horizon,
            self.replan_steps, self.control_frequency, self.device,
        )

    # ---- 构造 ----------------------------------------------------------------

    def _build(
        self,
        cfg: DictConfig,
        checkpoint_file: Path,
        stats_path: Path,
        model_dtype: torch.dtype,
    ) -> None:
        model = instantiate(cfg.model.model_arch)
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"checkpoint 里没有 `model_state_dict`: {checkpoint_file}")
        model = load_state_dict_safely(
            model, checkpoint["model_state_dict"], extra_prefixes=["normalizer."]
        )
        del checkpoint

        if model_dtype in (torch.bfloat16, torch.float16):
            model = model.to(model_dtype)
        if hasattr(model, "apply_fp32_params"):
            model.apply_fp32_params()
        self.policy = model.to(self.device).eval()
        if hasattr(self.policy, "action_tokenizer"):
            self.policy.action_tokenizer.to(self.device)

        processor = build_processors(cfg)

        # 选出目标 embodiment 的子 processor（14 维双臂）
        self._embodiment_key: Optional[str] = None
        if isinstance(processor, MixtureProcessor):
            available = sorted(processor.processors.keys())
            if self.embodiment in processor.processors:
                self._embodiment_key = self.embodiment
            elif len(processor.processors) == 1:
                self._embodiment_key = available[0]
            else:
                raise ValueError(
                    f"MixtureProcessor 里有多个 embodiment 但没有 `{self.embodiment}`: {available}"
                )
            self._sub_processor = processor[self._embodiment_key]
        else:
            self._sub_processor = processor
        self.processor = processor
        self.inferencer = PolicyInferencer(self.policy, processor, device=self.device)

        # ---- shape_meta：14 维布局的唯一真相来源 ----
        shape_meta = self._sub_processor.shape_meta
        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]
        self.state_dim = _flat_dim_from_meta(self.state_meta)
        self.action_dim = _flat_dim_from_meta(self.action_meta)

        image_keys = {str(m["key"]) for m in self.image_meta}
        missing = [k for k in CAMERA_KEYS if k not in image_keys]
        if missing:
            raise ValueError(
                f"processor 的 shape_meta['images'] 缺少相机: {missing}（有的是 {sorted(image_keys)}）"
            )

        self._preflight(cfg, stats_path)

        # 归一化统计量必须覆盖当前 embodiment，否则 MixtureProcessor 会抛 KeyError
        stats = load_dataset_stats_from_json(str(stats_path))
        try:
            processor.set_normalizer_from_stats(stats)
        except KeyError as exc:
            want = {str(m["key"]): _meta_dim(m) for m in self.action_meta}
            compatible = []
            for emb, block in stats.items():
                dims = {
                    k: (len(v["global_mean"]) if hasattr(v.get("global_mean"), "__len__") else None)
                    for k, v in block.get("action", {}).items()
                }
                if all(dims.get(k) == n for k, n in want.items()):
                    compatible.append(emb)
            raise RuntimeError(
                f"dataset_stats.json 里没有 embodiment `{self._embodiment_key}` 的统计量: {stats_path}\n"
                f"  原始报错: {exc}\n"
                f"  该 stats 覆盖的 embodiment: {sorted(stats)}\n"
                f"  其中与本 embodiment 动作布局 {want} 完全一致、可直接零样本复用的有:\n"
                f"    {compatible or '（无）'}\n"
                "  两条路:\n"
                "    a) 零样本: 把 deploy_policy.yml 的 sim_task / embodiment 换成上面某个名字\n"
                "       （需要有同名的 configs/data/<name>.yaml，可用 convert_lerobot_to_g05.py\n"
                "        的 --embodiment 参数生成）\n"
                "    b) 微调: 用 policy/g05/finetune.sh 在比赛数据上微调，\n"
                "       产物目录会自动生成配套的 dataset_stats.json"
            ) from exc
        processor.eval()

        self.num_obs_steps = int(self._sub_processor.num_obs_steps)
        if self.num_obs_steps <= 0:
            raise ValueError(f"num_obs_steps 必须为正，实际是 {self.num_obs_steps}")

        # 推理时不做训练期的动作过滤/截断，拿完整 chunk
        neutral_filter = BaseActionFilter()
        neutral_filter.set_shape_meta(self._sub_processor.shape_meta)
        self._sub_processor.action_filter = neutral_filter
        self._sub_processor.action_horizon = int(cfg.data.action_size)

        self.pending_actions: deque = deque()

    def _preflight(self, cfg: DictConfig, stats_path: Path) -> None:
        """提前拦住 checkpoint 与 embodiment 配置错配。

        cfg.model 整体来自 checkpoint 自带的 .hydra/config.yaml，所以 checkpoint 喂几路
        相机、动作多少维，全都写在里面。

        注意 `num_output_cameras` 不是"几台相机"，而是**图像槽位总数**
        = num_obs_steps × 相机数。官方 g05-base 是 `num_output_cameras: 18`
        / `num_obs_steps: 6` / `num_input_cameras: 3`，即 6 个观测步 × 3 路相机，
        相机数其实和比赛一致，别被 18 这个数字误导。
        """
        n_slots = OmegaConf.select(cfg, "model.processor.num_output_cameras")
        n_obs = int(OmegaConf.select(cfg, "model.processor.num_obs_steps") or 1)
        n_cameras = len(self.image_meta)
        expected = n_obs * n_cameras
        if n_slots is not None and int(n_slots) != expected:
            raise RuntimeError(
                f"checkpoint 的图像槽位数 num_output_cameras={int(n_slots)}，"
                f"但按本 embodiment 应为 num_obs_steps({n_obs}) × 相机数({n_cameras}) = {expected}。\n"
                f"  当前相机: {[str(m['key']) for m in self.image_meta]}\n"
                "  说明该 checkpoint 不是为这个 embodiment 训练的，请先微调再评估。"
            )

        action_dim = OmegaConf.select(cfg, "model.model_arch.action_dim")
        if action_dim is not None:
            logger.info(
                "模型内部动作宽度 action_dim=%s（分组布局），本 embodiment 打平后 %d 维；"
                "两者之间由 GroupedPaddingMerger 按 parts_meta 自动对齐",
                action_dim, self.action_dim,
            )

    # ---- 对外四方法 -----------------------------------------------------------

    def set_language(self, instruction: Optional[str]) -> None:
        self.instruction = instruction
        print(f"\nsuccessfully set instruction: {instruction}")

    def update_observation_window(self, img_dict: Dict[str, np.ndarray], state: np.ndarray) -> None:
        """img_dict: {cam_high/cam_left_wrist/cam_right_wrist: HWC uint8}; state: (14,)"""
        images: Dict[str, torch.Tensor] = {}
        for key in CAMERA_KEYS:
            if key not in img_dict:
                raise KeyError(f"观测缺少相机 `{key}`，实际有 {sorted(img_dict)}")
            chw = _as_chw_uint8(img_dict[key])
            if key not in self._logged_image_shapes:
                logger.info("相机 %s 输入 shape=%s", key, tuple(chw.shape))
                self._logged_image_shapes.add(key)
            # [T, 3, H, W]，num_obs_steps=1 时就是把当前帧复制一份
            images[key] = torch.from_numpy(chw).unsqueeze(0).expand(self.num_obs_steps, -1, -1, -1)

        state_vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(
                f"state 维度不对: 收到 {state_vector.shape[0]}，期望 {self.state_dim}"
            )

        self.observation_window = {
            "images": images,
            "state": _split_flat_vector(state_vector, self.state_meta, repeat_steps=self.num_obs_steps),
            "task": "" if self.instruction is None else str(self.instruction),
            "action": _zero_action_by_meta(self.action_meta, horizon=self.model_action_horizon),
            "action_is_pad": torch.ones(self.model_action_horizon, dtype=torch.bool),
            "state_is_pad": torch.zeros(self.num_obs_steps, dtype=torch.bool),
            "image_is_pad": torch.zeros(self.num_obs_steps, dtype=torch.bool),
            "idx": 0,
            "frequency": self.control_frequency,
        }
        if self._embodiment_key is not None:
            self.observation_window["embodiment"] = self._embodiment_key

    def get_action(self) -> np.ndarray:
        """跑一次推理，返回 [T, 14] 绝对关节位置，T = min(replan_steps, horizon)。"""
        if self.observation_window is None:
            raise RuntimeError("先调用 update_observation_window()")

        prev_cwd = Path.cwd()
        os.chdir(G05_ROOT)
        try:
            pred = self.inferencer.infer([self.observation_window])[0]
        finally:
            os.chdir(prev_cwd)

        chunk = self._assemble_chunk(pred)
        n_exec = min(self.replan_steps, self.exec_action_horizon, chunk.shape[0])
        if n_exec <= 0:
            raise RuntimeError("推理没有产出任何动作")
        return chunk[:n_exec]

    def reset_obsrvationwindows(self) -> None:
        self.instruction = None
        self.observation_window = None
        self.pending_actions.clear()
        print("successfully unset obs and language instruction")

    # ---- 输出解码 -------------------------------------------------------------

    def _assemble_chunk(self, pred: Dict[str, Any]) -> np.ndarray:
        """{left_arm:[T,6], left_gripper:[T,1], right_arm:[T,6], right_gripper:[T,1]} -> [T,14]

        按 shape_meta 的 start_index 回填，顺序与比赛 env 的
        [左臂6, 左夹爪1, 右臂6, 右夹爪1] 完全一致。
        """
        expected = [str(m["key"]) for m in self.action_meta]
        produced = [k for k in pred.keys() if not k.startswith("_")]
        missing = [k for k in expected if k not in pred]
        extra = sorted(set(produced) - set(expected))
        if missing or extra:
            raise ValueError(
                f"动作 key 对不上: 缺 {missing}, 多 {extra}, 实际产出 {sorted(produced)}"
            )

        chunk: Optional[np.ndarray] = None
        for meta in self.action_meta:
            key = str(meta["key"])
            value = pred[key]
            arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if arr.ndim == 3:
                if arr.shape[0] != 1:
                    raise ValueError(f"动作 `{key}` 的 batch 应为 1，实际 {arr.shape}")
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"动作 `{key}` 应为 [T,D] 或 [1,T,D]，实际 {tuple(arr.shape)}")

            dim = _meta_dim(meta)
            if arr.shape[1] != dim:
                raise ValueError(f"动作 `{key}` 维度不对: {arr.shape[1]} != {dim}")
            if chunk is None:
                chunk = np.zeros((arr.shape[0], self.action_dim), dtype=np.float32)
            elif chunk.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"动作 `{key}` 的时间步不一致: {arr.shape[0]} != {chunk.shape[0]}"
                )
            start = int(meta["start_index"])
            chunk[:, start : start + dim] = arr.astype(np.float32)

        if chunk is None:
            raise RuntimeError("推理没有产出任何动作")
        return chunk
