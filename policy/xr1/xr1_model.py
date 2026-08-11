# ----------------------------------------------------------------------------
# XR-1 (Xiaomi-Robotics-1) 模型封装 —— 进程内加载，不起 socket server
#
# 接口对齐 policy/pi05/pi_model.py:
#   set_language / update_observation_window / get_action / reset_obsrvationwindows
#
# 关键背景（详见 README_INTEGRATION.md）:
#   * XR-1 的动作是 (30, 60) 的“相对当前观测帧”的末端位姿增量，状态是 (1, 60) 的
#     纯关节+夹爪打包。本赛题机器人只有 14 维绝对关节位置，没有末端位姿，
#     因此动作侧走“槽位复用”：把每臂 6 个关节塞进 ee_pos(3) + ee_aa(3) 两个槽，
#     并用 exp/log 映射保证编解码严格可逆。
#   * 编码在 convert_lerobot_to_xr1.py 里完成，本文件做的是它的逆运算。
# ----------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# XR-1 动作打包槽位（与 mibot.utils.io.ACTION_PARTS 一致）
LEFT_POS_SLOT = slice(0, 3)
LEFT_AA_SLOT = slice(3, 6)
LEFT_GRIP_SLOT = 6
RIGHT_POS_SLOT = slice(8, 11)
RIGHT_AA_SLOT = slice(11, 14)
RIGHT_GRIP_SLOT = 14

# EmbodiChain 14 维 qpos 布局: [左臂6, 左夹爪1, 右臂6, 右夹爪1]
LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = 13

DEFAULT_GRIPPER_LIMITS = (0.0, 0.05)


def _ensure_xr1_importable(xr1_repo):
    """把 Xiaomi-Robotics-1/xr1 加进 sys.path，使 `import mibot` 可用。"""
    if xr1_repo and os.path.isdir(xr1_repo) and xr1_repo not in sys.path:
        sys.path.insert(0, xr1_repo)


def resolve_attn_implementation(preference="auto"):
    """flash-attn 装不上是常态，自动退回 sdpa。"""
    if preference and preference != "auto":
        return preference
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


@contextlib.contextmanager
def _patched_model_build(backbone_path, attn_implementation):
    """临时改写 mibot.models.VLA.XR1 的两个模块级依赖。

    XR-1 的 `_build_model()` 里写死了
        Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        Qwen3VLForConditionalGeneration._from_config(..., attn_implementation="flash_attention_2")
    我们既要指到本地 backbone 目录（离线机器上没法连 HF），又要能退回 sdpa。
    改上游源码会污染第三方仓库，所以在构建期打补丁、构建完立刻还原。
    """
    import mibot.models.VLA.XR1 as xr1_module

    original_config = xr1_module.Qwen3VLConfig
    original_vlm = xr1_module.Qwen3VLForConditionalGeneration

    class _ConfigShim:
        @staticmethod
        def from_pretrained(_name_or_path, **kwargs):
            return original_config.from_pretrained(backbone_path, **kwargs)

    class _VlmShim:
        @staticmethod
        def _from_config(config, **kwargs):
            kwargs["attn_implementation"] = attn_implementation
            return original_vlm._from_config(config, **kwargs)

    xr1_module.Qwen3VLConfig = _ConfigShim
    xr1_module.Qwen3VLForConditionalGeneration = _VlmShim
    try:
        yield
    finally:
        xr1_module.Qwen3VLConfig = original_config
        xr1_module.Qwen3VLForConditionalGeneration = original_vlm


def _strip_module_prefix(state_dict):
    stripped = {
        key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")
    }
    return stripped or dict(state_dict)


def _find_checkpoint_file(model_path):
    """定位权重文件，兼容三种形态。

    1. 直接给 .pt 文件
    2. 官方发布权重目录（内含 model_states.pt）
    3. 后训练输出目录（内含 last.ckpt/checkpoint/mp_rank_00_model_states.pt）
    """
    if os.path.isfile(model_path):
        return model_path

    candidates = [
        os.path.join(model_path, "model_states.pt"),
        os.path.join(model_path, "pretrained_ckpt", "model_states.pt"),
        os.path.join(model_path, "last.ckpt", "checkpoint", "mp_rank_00_model_states.pt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"在 {model_path} 下找不到 XR-1 权重。已尝试: {candidates}"
    )


class Stats:
    """打包动作的 per-step mean/std 与打包状态的 q01/q99。

    XR-1 的模型本身只吐归一化动作，反归一化统计量存在训练配置里
    （见 mibot/server/deploy.py::load_stats），所以部署侧必须自己带一份。
    """

    def __init__(self, mean, std, q01, q99, rot_scale=1.0, source="unknown",
                 encoding="slot", gripper_range=None):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        self.rot_scale = float(rot_scale)
        self.source = source
        # 编码方式决定解码方式：eef 必须走 IK，slot 直接加到关节上
        self.encoding = encoding
        # 夹爪量程随数据集而变（sample_loading 是 0~1 归一化，不是 0~0.05 米），
        # 硬编码会把夹爪指令裁没
        self.gripper_range = tuple(gripper_range) if gripper_range else None

        if self.mean.shape != self.std.shape or self.mean.ndim != 2:
            raise ValueError(f"mean/std 形状不合法: {self.mean.shape} vs {self.std.shape}")
        if self.q01.shape != (1, 60) or self.q99.shape != (1, 60):
            raise ValueError(f"q01/q99 形状必须是 (1, 60)，当前 {self.q01.shape} / {self.q99.shape}")

    @property
    def action_length(self):
        return int(self.mean.shape[0])

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as handle:
            payload = json.load(handle)
        return cls(
            mean=payload["mean"],
            std=payload["std"],
            q01=payload["q01"],
            q99=payload["q99"],
            rot_scale=payload.get("rot_scale", 1.0),
            source=path,
            encoding=payload.get("encoding", "slot"),
            gripper_range=payload.get("gripper_range"),
        )

    @classmethod
    def from_hydra_yaml(cls, path):
        """从 xr1/configs/data/*.yaml 里读统计量（load_washer.yaml 就是这个格式）。"""
        import yaml

        with open(path, "r") as handle:
            payload = yaml.safe_load(handle)
        train = payload["data"]["params"]["train_datasets"]
        return cls(
            mean=train["mean"],
            std=train["std"],
            q01=train["q01"],
            q99=train["q99"],
            rot_scale=float(train.get("rot_scale", 1.0)),
            source=path,
        )

    @classmethod
    def identity(cls, action_length=30):
        """兜底：不做任何反归一化。形状对但语义无意义，仅用于跑通链路。"""
        return cls(
            mean=np.zeros((action_length, 60), dtype=np.float32),
            std=np.ones((action_length, 60), dtype=np.float32),
            q01=np.full((1, 60), -1.0, dtype=np.float32),
            q99=np.full((1, 60), 1.0, dtype=np.float32),
            rot_scale=1.0,
            source="identity(fallback)",
        )


def load_stats(stats_path=None, xr1_repo=None, action_length=30):
    if stats_path and os.path.isfile(stats_path):
        if stats_path.endswith((".yaml", ".yml")):
            return Stats.from_hydra_yaml(stats_path)
        return Stats.from_json(stats_path)

    if stats_path:
        print(f"[XR1] 警告: stats_path 不存在: {stats_path}")

    if xr1_repo:
        fallback = os.path.join(xr1_repo, "configs", "data", "load_washer.yaml")
        if os.path.isfile(fallback):
            print(f"[XR1] 警告: 未提供统计量，回退到官方 demo 配置 {fallback}（零样本用，语义不匹配本赛题）")
            return Stats.from_hydra_yaml(fallback)

    print("[XR1] 警告: 找不到任何统计量，使用恒等反归一化（仅能验证形状）")
    return Stats.identity(action_length)


class XR1:
    """XR-1 策略的进程内封装。"""

    def __init__(
        self,
        model_path,
        backbone_path,
        xr1_repo,
        stats_path=None,
        xr1_step=10,
        device="cuda",
        attn_implementation="auto",
        gripper_limits=None,
        max_joint_delta=None,
        decode_mode="auto",
        ik_shrink_retry=(0.5, 0.25, 0.1),
    ):
        _ensure_xr1_importable(xr1_repo)

        self.xr1_repo = xr1_repo
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.xr1_step = int(xr1_step)
        self.gripper_limits = gripper_limits
        self.max_joint_delta = max_joint_delta
        self.decode_mode = decode_mode
        # IK 解不出来时，把目标按这些比例向锚点位姿收缩后重试。
        # 锚点位姿必然可达（它就是当前构型的 FK），所以收缩总能找到落点。
        # 实测（10 集、目标误差 0.03 量级）：失败率 20.43% -> 5.15%。
        # 另外两个方案实测无效：改锚点种子重试 20.43%（说明不是种子/分支问题），
        # SVD 重正交化 20.43%（解码出的旋转本来就正交，没有可修的东西）。
        # 设成空元组即可关闭。
        self.ik_shrink_retry = tuple(ik_shrink_retry or ())
        self._robot = None
        self.ik_shrink_rescues = 0
        self.ik_calls = 0
        self.ik_failures = 0
        self.attn_implementation = resolve_attn_implementation(attn_implementation)

        from mibot.utils.io import build_action_mask

        self.stats = load_stats(stats_path, xr1_repo)
        self.action_length = self.stats.action_length
        if self.xr1_step > self.action_length:
            print(
                f"[XR1] 警告: xr1_step={self.xr1_step} 超过动作horizon={self.action_length}，已截断"
            )
            self.xr1_step = self.action_length

        self.model = self._build_model(model_path, backbone_path)
        self.processor = self._build_processor(backbone_path)

        action_mask = build_action_mask(self.action_length)
        self._action_mask = torch.from_numpy(action_mask).to(self.device)[None]
        self._mean = torch.from_numpy(self.stats.mean).to(self.device)[None]
        self._std = torch.from_numpy(self.stats.std).to(self.device)[None]
        self._q01 = torch.from_numpy(self.stats.q01).to(self.device)
        self._q99 = torch.from_numpy(self.stats.q99).to(self.device)

        # 解码方式默认跟随训练时的编码方式，避免手滑配错
        if self.decode_mode == "auto":
            self.decode_mode = "eef_ik" if self.stats.encoding == "eef" else "slot"
        if self.gripper_limits is None:
            self.gripper_limits = self.stats.gripper_range or DEFAULT_GRIPPER_LIMITS

        self.instruction = None
        self.observation_window = None

        print(
            f"[XR1] 就绪: device={self.device} attn={self.attn_implementation} "
            f"horizon={self.action_length} exec_steps={self.xr1_step} "
            f"stats={self.stats.source} rot_scale={self.stats.rot_scale:.4f} "
            f"encoding={self.stats.encoding} decode={self.decode_mode} "
            f"gripper_limits={self.gripper_limits}"
        )

    # ---------------------------------------------------------------- 构建

    def _build_model(self, model_path, backbone_path):
        config_py = os.path.join(model_path, "config.py") if os.path.isdir(model_path) else None
        if config_py and os.path.isfile(config_py):
            return self._build_from_training_output(model_path, backbone_path)
        if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
            return self._build_from_hf(model_path)
        return self._build_from_state_dict(model_path, backbone_path, model_config=None)

    def _build_from_hf(self, model_path):
        """官方 deploy/server.py 的路径。当前 HF 发布的 Xiaomi-Robotics-1-5B 只有裸
        model_states.pt、没有 config.json，所以这条分支目前走不到，留给将来的 HF 格式发布。"""
        from transformers import AutoModel

        print(f"[XR1] 以 HF 格式加载: {model_path}")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            dtype=torch.bfloat16,
        )
        self._is_hf_model = True
        return model.eval().to(self.device)

    def _build_from_training_output(self, model_path, backbone_path):
        """后训练输出目录：config.py 决定模型超参，权重在 last.ckpt 下。"""
        from mmengine import Config

        cfg = Config.fromfile(os.path.join(model_path, "config.py"))
        model_config = dict(cfg.model.params.model)
        # 训练配置里若带自定义统计量，优先于外部 stats
        try:
            train_datasets = cfg.data.params.train_datasets
            self.stats = Stats(
                mean=train_datasets.mean,
                std=train_datasets.std,
                q01=train_datasets.q01,
                q99=train_datasets.q99,
                rot_scale=float(train_datasets.get("rot_scale", self.stats.rot_scale)),
                source=os.path.join(model_path, "config.py"),
                # 老的训练配置没写 encoding（转换器后来才补上），此时沿用外部
                # stats 的值；两边都没有才退回 slot。配错解码器会静默产出垃圾动作，
                # 所以这里宁可继承也不要盲目默认。
                encoding=train_datasets.get("encoding", self.stats.encoding),
                gripper_range=train_datasets.get("gripper_range", self.stats.gripper_range),
            )
            self.action_length = self.stats.action_length
            self.xr1_step = min(self.xr1_step, self.action_length)
            print(f"[XR1] 已从训练配置读取统计量: {self.stats.source}")
        except Exception as exc:  # 配置里没有 data 段就保持外部 stats
            print(f"[XR1] 训练配置中没有可用统计量，沿用 {self.stats.source} ({exc})")
        return self._build_from_state_dict(model_path, backbone_path, model_config)

    def _build_from_state_dict(self, model_path, backbone_path, model_config):
        from mibot.models import MIMODEL

        checkpoint_path = _find_checkpoint_file(model_path)
        config = dict(model_config or {})
        config.setdefault("type", "xr1")
        # 训练期开关，推理不需要
        config["ffn_gradient_checkpointing"] = False

        print(f"[XR1] 构建骨架 (backbone={backbone_path}, attn={self.attn_implementation})")
        with _patched_model_build(backbone_path, self.attn_implementation):
            model = MIMODEL.build(config)

        print(f"[XR1] 加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        state_dict = checkpoint.get("module", checkpoint)
        info = model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
        print(f"[XR1] load_state_dict: {info}")

        self._is_hf_model = False
        return model.eval().to(torch.bfloat16).to(self.device)

    def _build_processor(self, backbone_path):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(backbone_path)
        processor.tokenizer.padding_side = "right"
        return processor

    # ---------------------------------------------------------------- 接口

    def bind_env(self, env):
        """eef_ik 解码要用 EmbodiChain 的 FK/IK，这里拿到 robot 句柄。"""
        if self._robot is not None:
            return
        robot = getattr(getattr(env, "unwrapped", env), "robot", None)
        if robot is None:
            raise RuntimeError(
                "拿不到 env.unwrapped.robot，eef_ik 解码需要它做 FK/IK；"
                "若只想跑通形状可改用 decode_mode=slot"
            )
        self._robot = robot

    def ik_failure_rate(self):
        if self.ik_calls == 0:
            return 0.0
        return self.ik_failures / self.ik_calls

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"[XR1] instruction = {instruction!r}")

    def update_observation_window(self, img_arr, state):
        """img_arr = [cam_high, cam_right_wrist, cam_left_wrist]（与 pi05 的顺序一致）。"""
        image_high, image_right, image_left = img_arr[0], img_arr[1], img_arr[2]
        self.observation_window = {
            "ego": self._to_pil(image_high),
            "wrist_left": self._to_pil(image_left),
            "wrist_right": self._to_pil(image_right),
            "qpos": np.asarray(state, dtype=np.float32).reshape(-1),
        }

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        if self.ik_calls:
            print(
                f"[XR1] 上一集 IK 失败率 {self.ik_failures}/{self.ik_calls} "
                f"= {100.0 * self.ik_failure_rate():.2f}%"
                f"（收缩重试救回 {self.ik_shrink_rescues} 次）"
            )
        self.ik_calls = 0
        self.ik_failures = 0
        self.ik_shrink_rescues = 0
        print("[XR1] observation window / instruction 已清空")

    @torch.no_grad()
    def get_action(self):
        assert self.observation_window is not None, "先调用 update_observation_window!"

        from mibot.utils.io import denormalize_action

        qpos = self.observation_window["qpos"]
        if qpos.shape[0] != 14:
            raise ValueError(f"期望 14 维 qpos，实际 {qpos.shape}")

        batch = self._build_batch(qpos)

        if self._is_hf_model:
            outputs = self.model(**batch)
            action = outputs.actions
        else:
            action = self.model.generate(batch)

        action = action.float()
        mask = self._action_mask.float()
        action = denormalize_action(action * mask, self._mean, self._std) * mask
        packed = action[0].cpu().numpy().astype(np.float32)

        if self.decode_mode == "eef_ik":
            return self.decode_action_eef_ik(packed, qpos)
        return self.decode_action(packed, qpos)

    # ------------------------------------------------------------ 前处理

    @staticmethod
    def _to_pil(image):
        array = np.asarray(image)
        if array.ndim == 4:  # (1, H, W, C)
            array = array[0]
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            maximum = float(np.nanmax(array)) if array.size else 0.0
            if maximum <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(array))

    @staticmethod
    def _messages(instruction, ego, wrist_left, wrist_right):
        """与 mibot/server/runtime/client.py::_messages 逐字对齐，不能改动措辞。"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The following observations are captured from multiple views.\n# Ego View\n",
                    },
                    {"type": "image", "image": ego},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": wrist_left},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": wrist_right},
                    {
                        "type": "text",
                        "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot",
                    },
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def _build_batch(self, qpos):
        from mibot.utils.io import ACTION_EPS, compose_state, resize_image

        window = self.observation_window
        images = [
            resize_image(window[key], factor=32, max_pixels=160000)
            for key in ("ego", "wrist_left", "wrist_right")
        ]

        payload = self.processor.apply_chat_template(
            [self._messages(self.instruction, *images)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )

        batch = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in payload.items()
        }

        state = compose_state(
            left_gripper=qpos[LEFT_GRIPPER],
            left_joint=qpos[LEFT_ARM],
            right_gripper=qpos[RIGHT_GRIPPER],
            right_joint=qpos[RIGHT_ARM],
        )
        state = torch.from_numpy(state).to(self.device)[None]

        # 与 runtime/server.py 一致的分位数归一化
        valid = self._q99 > self._q01
        normalized = torch.zeros_like(state)
        normalized[..., valid[0]] = (
            2.0
            * (state[..., valid[0]] - self._q01[..., valid[0]])
            / (self._q99[..., valid[0]] - self._q01[..., valid[0]] + ACTION_EPS)
            - 1.0
        )
        batch["state"] = normalized.clamp(-1.0, 1.0).to(torch.bfloat16)
        batch["action"] = torch.zeros(
            (1, self.action_length, 60), device=self.device, dtype=torch.bfloat16
        )
        batch["action_mask"] = self._action_mask
        return batch

    # ------------------------------------------------------------ 后处理

    def decode_action_eef_ik(self, packed, anchor_qpos):
        """EEF 编码的解码：末端增量 -> 绝对末端目标 -> IK -> 14 维绝对 qpos。

        与训练侧编码严格互逆（见 convert_lerobot_to_xr1.py 的 eef 分支）：
            p_t = p_a + R_a · dims[0:3]
            R_t = R_a · exp(dims[3:6])
        其中 (p_a, R_a) 是当前 qpos 的末端位姿。

        坐标系：全程走 EmbodiChain 的 arena 系（compute_fk 出 arena、compute_ik 收
        arena），自洽，不需要任何外参。训练数据用臂基座系算是等价的——增量对常量
        刚体变换左乘不变（validate_fk.py 实测 1.5e-15）。

        IK 失败策略：保持上一步的关节角并计数；种子始终用上一步的解，保证轨迹连续。
        """
        if self._robot is None:
            raise RuntimeError("请先调用 bind_env(env)")

        packed = np.asarray(packed, dtype=np.float32)
        anchor = np.asarray(anchor_qpos, dtype=np.float32).reshape(-1)
        steps = packed.shape[0]
        targets = np.tile(anchor[None], (steps, 1)).astype(np.float32)

        from mibot.utils.io import aa2rotm

        device = getattr(self._robot, "device", "cpu")
        sides = (
            ("left_arm", LEFT_ARM, LEFT_GRIPPER, LEFT_POS_SLOT, LEFT_AA_SLOT, LEFT_GRIP_SLOT),
            ("right_arm", RIGHT_ARM, RIGHT_GRIPPER, RIGHT_POS_SLOT, RIGHT_AA_SLOT, RIGHT_GRIP_SLOT),
        )

        for part, arm_slice, gripper_index, pos_slot, aa_slot, grip_slot in sides:
            seed = torch.as_tensor(anchor[arm_slice], dtype=torch.float32, device=device)[None]

            pose = self._robot.compute_fk(seed, name=part, to_matrix=True)
            pose = pose[0].detach().cpu().numpy().astype(np.float64)
            anchor_rotation = pose[:3, :3]
            anchor_position = pose[:3, 3]

            # 30 步共用同一个锚点（上游是广播差分，不是逐步累积）
            positions = anchor_position[None] + packed[:, pos_slot] @ anchor_rotation.T
            rotations = np.stack(
                [anchor_rotation @ aa2rotm(delta) for delta in packed[:, aa_slot]], axis=0
            )

            last = seed
            for step in range(steps):
                target = np.eye(4, dtype=np.float32)
                target[:3, :3] = rotations[step]
                target[:3, 3] = positions[step]

                def try_ik(pose_matrix):
                    tensor = torch.as_tensor(pose_matrix, dtype=torch.float32, device=device)[None]
                    outcome = self._robot.compute_ik(pose=tensor, joint_seed=last, name=part)
                    if outcome is None:
                        return None
                    code, solution = outcome
                    if int(np.asarray(code.detach().cpu()).reshape(-1)[0]) != 1:
                        return None
                    return solution.reshape(1, -1)[:, : arm_slice.stop - arm_slice.start]

                self.ik_calls += 1
                solved = try_ik(target)

                # 主要失败模式是目标落在可达集外（不是种子选错），
                # 所以向必然可达的锚点收缩重试
                if solved is None and self.ik_shrink_retry:
                    for alpha in self.ik_shrink_retry:
                        shrunk = np.eye(4, dtype=np.float32)
                        shrunk[:3, 3] = anchor_position + alpha * (positions[step] - anchor_position)
                        rotation = anchor_rotation + alpha * (rotations[step] - anchor_rotation)
                        u, _, vt = np.linalg.svd(rotation)
                        rotation = u @ vt
                        if np.linalg.det(rotation) < 0:
                            u[:, -1] *= -1
                            rotation = u @ vt
                        shrunk[:3, :3] = rotation
                        solved = try_ik(shrunk)
                        if solved is not None:
                            self.ik_shrink_rescues += 1
                            break

                if solved is not None:
                    last = solved
                else:
                    self.ik_failures += 1  # 收缩也解不出来才保持上一步
                targets[step, arm_slice] = last[0].detach().cpu().numpy()

            targets[:, gripper_index] = anchor[gripper_index] + packed[:, grip_slot]

        if self.gripper_limits is not None:
            low, high = self.gripper_limits
            targets[:, LEFT_GRIPPER] = np.clip(targets[:, LEFT_GRIPPER], low, high)
            targets[:, RIGHT_GRIPPER] = np.clip(targets[:, RIGHT_GRIPPER], low, high)

        return targets.astype(np.float32)

    def decode_action(self, packed, anchor_qpos):
        """把 (T, 60) 的打包相对动作还原成 (T, 14) 的绝对 qpos。

        这是 convert_lerobot_to_xr1.py 里编码的严格逆运算：
            dims[0:3]  = R_anchor^T @ (q123_target - q123_anchor)
            dims[3:6]  = log(R_anchor^T @ exp(s * q456_target))
            dims[6]    = gripper_target - gripper_anchor
        其中 R_anchor = exp(s * q456_anchor)，s = rot_scale。
        """
        from mibot.utils.io import aa2rotm, rotm2aa_batch

        packed = np.asarray(packed, dtype=np.float32)
        anchor = np.asarray(anchor_qpos, dtype=np.float32).reshape(-1)
        scale = self.stats.rot_scale
        steps = packed.shape[0]
        targets = np.tile(anchor[None], (steps, 1)).astype(np.float32)

        sides = (
            (LEFT_ARM, LEFT_GRIPPER, LEFT_POS_SLOT, LEFT_AA_SLOT, LEFT_GRIP_SLOT),
            (RIGHT_ARM, RIGHT_GRIPPER, RIGHT_POS_SLOT, RIGHT_AA_SLOT, RIGHT_GRIP_SLOT),
        )

        for arm_slice, gripper_index, pos_slot, aa_slot, grip_slot in sides:
            arm_anchor = anchor[arm_slice]
            rotation_anchor = aa2rotm(scale * arm_anchor[3:6])

            # 位置槽：增量表达在 anchor 旋转坐标系下，左乘 R_anchor 还原
            delta_first = packed[:, pos_slot] @ rotation_anchor.T
            targets[:, arm_slice.start : arm_slice.start + 3] = arm_anchor[None, 0:3] + delta_first

            # 旋转槽：R_target = R_anchor @ exp(delta)，再 log 回关节三元组
            rotations = np.stack(
                [rotation_anchor @ aa2rotm(delta) for delta in packed[:, aa_slot]], axis=0
            )
            targets[:, arm_slice.start + 3 : arm_slice.start + 6] = (
                rotm2aa_batch(rotations) / scale
            )

            targets[:, gripper_index] = anchor[gripper_index] + packed[:, grip_slot]

        if self.max_joint_delta is not None:
            limit = float(self.max_joint_delta)
            for arm_slice in (LEFT_ARM, RIGHT_ARM):
                delta = np.clip(targets[:, arm_slice] - anchor[None, arm_slice], -limit, limit)
                targets[:, arm_slice] = anchor[None, arm_slice] + delta

        if self.gripper_limits is not None:
            low, high = self.gripper_limits
            targets[:, LEFT_GRIPPER] = np.clip(targets[:, LEFT_GRIPPER], low, high)
            targets[:, RIGHT_GRIPPER] = np.clip(targets[:, RIGHT_GRIPPER], low, high)

        return targets.astype(np.float32)
