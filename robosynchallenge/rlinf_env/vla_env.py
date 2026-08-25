# ----------------------------------------------------------------------------
# 把 RoboSynChallenge 的任务接进 RLinf 的具身 RL 栈(PPO / GRPO)。
#
# RLinf 已经有一个 EmbodiChain 适配器,但它是为 CartPole 写的:观测只有状态向量,
# reward 直接取环境返回值。VLA 后训练需要两样它没有的东西:
#
#   1. 图像 + 语言。RLinf 的 actor 吃的观测契约见
#      RLinf/rlinf/data/schema/embodied_types.py:111-117 ——
#         main_images   [N_ENV, H, W, C]
#         wrist_images  [N_ENV, H, W, C] 或 [N_ENV, N_IMG, H, W, C]   <- 双腕走后者
#         states        [N_ENV, D]
#         task_descriptions  list[str],每个 env 一条
#
#   2. 非零 reward。RoboSynChallenge 的任务没实现 get_reward,基类默认返回全 0;
#      而且为了数据采集,它们的 compute_task_state 故意把 success 压成全 False 让
#      episode 跑满,真实成败只由官方 is_task_success() 给出。
#
# 判定标准一律用官方 is_task_success(),不改写、不 shaping。见 install_official_reward。
# ----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Optional

import torch

from rlinf.envs.embodichain.embodichain_env import EmbodiChainEnv, _cfg_get

__all__ = ["RoboSynChallengeVLAEnv", "install_official_reward"]

# EmbodiChain 相机观测的键形如 obs["sensor"][<uid>]["color"],张量是 [N, H, W, C],
# C 可能是 4(RGBA)。这些默认值对应 configs/<task>/random/gym_config.json 的三路相机。
DEFAULT_MAIN_CAMERA = "cam_high"
DEFAULT_WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")

# 本仓库扩展了 EmbodiChain 的 manager 体系(自定义 action / dataset / event / observation
# 的 Functor),gym_config 里按函数名引用它们,例如 mixer_operating 的
# randomize_distractor_slots -> robosynchallenge.managers.events 里的
# replace_distractor_slots_from_library。
#
# 这些模块不会随 `import robosynchallenge` 自动注册——scripts/eval_policy.py 是把它们
# 直接塞进 DEFAULT_MANAGER_MODULES 的。RLinf 的 _build_env 只调 get_manager_modules(),
# 不注册就会在建环境时报 "Function '...' not found"。
CHALLENGE_MANAGER_MODULES = (
    "robosynchallenge.managers.actions",
    "robosynchallenge.managers.datasets",
    "robosynchallenge.managers.events",
    "robosynchallenge.managers.observations",
)


def install_official_reward(
    env,
    success_reward: float = 1.0,
    terminate_on_success: bool = True,
) -> None:
    """给已建好的 EmbodiChain 环境实例挂上官方判定驱动的奖励与终止。

    只改实例,不动 robosynchallenge/tasks/ 下的任何文件——那些是官方赛题代码,
    改了会和上游冲突,而且 12 个任务逐个改子类既冗余又容易漏。

    做两件事:

    ``compute_task_state``
        原实现返回的 success 常年为 False(任务刻意让 episode 跑满 500 步以便采集
        完整轨迹)。这里改成返回官方 ``is_task_success()``,于是
        ``base_env.step()`` 里的 ``terminateds = success | fail`` 就能在成功当步结束
        episode。对 PPO 这既省仿真步数,又缩短了 credit assignment 的跨度。

    ``get_reward``
        基类默认返回全 0。这里给稀疏奖励:成功那一步给 ``success_reward``,其余 0。
        注意 ``base_env.step()`` 是先 ``get_info()`` 再 ``get_reward(info=info)``,
        所以这里读到的 ``info["success"]`` 已经是上面替换过的官方判定。

    关于重复调用的安全性:原 ``compute_task_state`` 和 ``is_task_success`` 都可能触碰
    有副作用的统计(如 mixer_operating 的 ``_update_button_contact_history``)。官方
    实现本身就假定 ``is_task_success`` 会被反复调用并做了防重复计数——例如
    handle_basket 用 ``_hb_last_success_check_env_step`` 按环境步差累加而不是每次 +1。
    官方评测循环也同样在每步之外额外调用它。所以这里保持原调用不变、额外调一次官方
    判定,是被官方代码支持的用法。
    """
    unwrapped = env.unwrapped
    original_compute_task_state = unwrapped.compute_task_state

    def compute_task_state(**kwargs):
        success, fail, metrics = original_compute_task_state(**kwargs)
        official = unwrapped.is_task_success(**kwargs)
        official = torch.as_tensor(official, device=success.device).reshape(-1).to(torch.bool)

        metrics = dict(metrics) if metrics else {}
        metrics["official_success"] = official
        # 保留任务原本的 success 以便排查两者不一致的情况
        metrics["task_reported_success"] = success.to(torch.bool)

        if terminate_on_success:
            success = official
        return success, fail, metrics

    def get_reward(obs=None, action=None, info=None, **kwargs) -> torch.Tensor:
        if info is not None and "success" in info:
            success = info["success"]
        else:  # get_reward 早于 get_info 被调用时的兜底,正常路径走不到
            success = unwrapped.is_task_success()
        success = torch.as_tensor(success, device=unwrapped.device).reshape(-1)
        return success.to(torch.float32) * float(success_reward)

    unwrapped.compute_task_state = compute_task_state
    unwrapped.get_reward = get_reward


class RoboSynChallengeVLAEnv(EmbodiChainEnv):
    """EmbodiChain 环境 + 多路相机 + 语言指令 + 官方判定奖励。

    额外读取的 env 配置字段(都有默认值):

    ``main_camera``          主视角相机 uid,默认 ``cam_high``
    ``wrist_cameras``        腕部相机 uid 列表,默认 ``[cam_left_wrist, cam_right_wrist]``
    ``image_size``           缩放到的边长,默认 224(pi0.5 的输入尺寸)。设为 0 表示不缩放。
                             原始是 640x480,不缩放的话每步要在进程间搬 4.6 倍的数据。
    ``task_description``     语言指令。不填则从 gym_config 里取。
    ``success_reward``       成功奖励,默认 1.0
    ``terminate_on_success`` 成功即终止,默认 True
    ``state_key``            状态取自 ``obs["robot"][<key>]``,默认 ``qpos``
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ):
        # 这些必须在 super().__init__ 之前设好:父类构造过程里会建环境并调 _wrap_obs。
        self.main_camera = str(_cfg_get(cfg, "main_camera", DEFAULT_MAIN_CAMERA))
        wrist = _cfg_get(cfg, "wrist_cameras", list(DEFAULT_WRIST_CAMERAS))
        self.wrist_cameras = [str(x) for x in wrist]
        self.image_size = int(_cfg_get(cfg, "image_size", 224))
        self.state_key = str(_cfg_get(cfg, "state_key", "qpos"))
        self.success_reward = float(_cfg_get(cfg, "success_reward", 1.0))
        self.terminate_on_success = bool(_cfg_get(cfg, "terminate_on_success", True))
        self._configured_description = _cfg_get(cfg, "task_description", None)
        self._gym_config: Optional[dict] = None

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        self.task_description = self._resolve_task_description()
        self._task_descriptions = [self.task_description] * self.num_envs

    # -- 环境构建 ---------------------------------------------------------

    def _build_env(self):
        """重写而非调用父类:需要在 config_to_cfg 之前动 gym_config,而父类是在方法内部
        加载并立即消费它的,没有留插入点。逻辑与
        ``rlinf/envs/embodichain/embodichain_env.py`` 的同名方法一致,只多两处改动
        (见下面的注释)。"""
        from copy import deepcopy

        from embodichain.lab.gym.utils.gym_utils import (
            config_to_cfg,
            get_manager_modules,
            register_manager_modules,
        )
        from embodichain.lab.gym.utils.registration import (
            build_env,
            discover_task_packages,
            execute_init_hooks,
        )
        from embodichain.lab.sim import SimulationManagerCfg
        from embodichain.utils.utility import load_config

        from rlinf.envs.embodichain.embodichain_env import _resolve_gym_config_path

        # RoboSynChallenge 的任务靠 @register_env 在 import 时注册,但
        # discover_task_packages() 只会导入声明了 "embodichain.tasks" entry point 的包,
        # 而本仓库的 pyproject.toml 没有声明。Ray 会在独立进程里起 env worker,
        # 所以这个 import 必须发生在 worker 进程内,不能只在主进程做一次。
        import robosynchallenge  # noqa: F401

        # 改动 1:注册本仓库的 manager 模块。gym_config 里按函数名引用它们
        # (如 randomize_distractor_slots),不注册会报 "Function '...' not found"。
        register_manager_modules(list(CHALLENGE_MANAGER_MODULES))

        gym_config_path_cfg = _cfg_get(self.cfg, "gym_config_path")
        if not gym_config_path_cfg:
            raise ValueError("env 配置里缺 gym_config_path。")

        discover_task_packages()
        execute_init_hooks()

        gym_config_path_str = str(gym_config_path_cfg)
        if gym_config_path_str.startswith("embodichain_tasks/"):
            gym_config = load_config(gym_config_path_str)
        else:
            gym_config = load_config(str(_resolve_gym_config_path(gym_config_path_str)))

        # 指令写在 gym_config 里,父类不保留它,这里留一份给 _resolve_task_description。
        self._gym_config = deepcopy(gym_config)

        # 改动 2:剥掉 dataset 段。它是给数据采集用的 LeRobot 录制器,RL rollout 不需要,
        # 留着有两个问题:每个并行环境都会往磁盘写轨迹(纯浪费),而且它调的
        # LeRobotDataset.create(metadata_buffer_size=...) 只有 lerobot 0.4.x 有——
        # RLinf 的 rlinf-openpi 会把 lerobot 降到 0.3.3,于是建环境时直接 RuntimeError。
        env_section = gym_config.get("env")
        if isinstance(env_section, dict) and "dataset" in env_section:
            env_section = dict(env_section)
            env_section.pop("dataset", None)
            gym_config = dict(gym_config)
            gym_config["env"] = env_section

        env_cfg = config_to_cfg(
            deepcopy(gym_config), manager_modules=get_manager_modules()
        )
        env_cfg.num_envs = self.num_envs
        env_cfg.max_episode_steps = self.max_episode_steps
        env_cfg.sim_cfg = SimulationManagerCfg(
            headless=bool(_cfg_get(self.cfg, "headless", True)),
            sim_device=self._sim_device,
            gpu_id=self._gpu_id,
        )
        env = build_env(gym_config["id"], base_env_cfg=env_cfg)

        install_official_reward(
            env,
            success_reward=self.success_reward,
            terminate_on_success=self.terminate_on_success,
        )
        return env

    @staticmethod
    def _instruction_to_text(value: Any) -> Optional[str]:
        """指令可能是纯字符串,也可能是 {"lang": "..."} 这样的多语言字典。"""
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for key in ("lang", "en", "text"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            for text in value.values():
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return None

    def _resolve_task_description(self) -> str:
        if self._configured_description:
            return str(self._configured_description)

        cfg = self._gym_config or {}

        # 指令写在 dataset 录制器的参数里:env.dataset.<functor>.params.instruction,
        # 值形如 {"lang": "Pick the beaker, ..."}。规则与
        # scripts/eval_policy.py 的 extract_instruction_from_gym_config 一致。
        #
        # 注意 _build_env 会为 RL 剥掉 dataset 段(见那里的注释),所以这里读的是
        # self._gym_config —— 剥之前留的那份副本。顺序反了就会拿不到指令。
        dataset_cfg = (cfg.get("env") or {}).get("dataset") or {}
        if isinstance(dataset_cfg, dict):
            for functor_cfg in dataset_cfg.values():
                if not isinstance(functor_cfg, dict):
                    continue
                text = self._instruction_to_text(
                    (functor_cfg.get("params") or {}).get("instruction")
                )
                if text:
                    return text

        # 兜底:少数配置可能把指令放在顶层或 env 下
        for holder in (cfg.get("env") or {}, cfg):
            if not isinstance(holder, dict):
                continue
            for key in ("instruction", "task_description", "prompt"):
                text = self._instruction_to_text(holder.get(key))
                if text:
                    return text

        raise ValueError(
            "找不到任务指令。查过 env.dataset.<functor>.params.instruction 和顶层的 "
            "instruction/task_description/prompt 都没有。请在 env 配置里显式给 "
            "task_description —— pi0.5 的 prompt 不能为空,空指令会让策略行为与 SFT 时完全不同。"
        )

    # -- 观测 -------------------------------------------------------------

    def _camera_image(self, raw_obs: dict, uid: str) -> torch.Tensor:
        """取一路相机,返回 [N, H, W, 3] 的 uint8。"""
        sensor = raw_obs.get("sensor")
        if sensor is None or uid not in sensor.keys():
            available = list(sensor.keys()) if sensor is not None else []
            raise KeyError(
                f"gym_config 里没有相机 {uid!r}。可用的: {available}。"
                "检查 env 配置的 main_camera / wrist_cameras 是否和 "
                "configs/<task>/<setting>/gym_config.json 的 sensor uid 对得上。"
            )

        image = sensor[uid]["color"]
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim != 4:
            raise ValueError(f"相机 {uid} 的图像应是 [N, H, W, C],实际 {tuple(image.shape)}")

        image = image[..., :3]  # EmbodiChain 给的是 RGBA,pi0.5 只要 RGB
        if image.dtype != torch.uint8:
            # 浮点图按 0-1 归一化保存,先还原到 0-255 再统一成 uint8
            image = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        if self.image_size and image.shape[1] != self.image_size:
            # interpolate 要 NCHW 浮点;缩放在 env 侧做,避免把 640x480 原图跨进程搬给 actor
            chw = image.permute(0, 3, 1, 2).to(torch.float32)
            chw = torch.nn.functional.interpolate(
                chw,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            image = chw.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        return image.contiguous()

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        main_images = self._camera_image(raw_obs, self.main_camera)

        wrist_list = [self._camera_image(raw_obs, uid) for uid in self.wrist_cameras]
        if len(wrist_list) == 1:
            wrist_images = wrist_list[0]  # [N, H, W, C]
        else:
            wrist_images = torch.stack(wrist_list, dim=1)  # [N, N_IMG, H, W, C]

        robot_obs = raw_obs["robot"]
        if self.state_key not in robot_obs.keys():
            raise KeyError(
                f"obs['robot'] 里没有 {self.state_key!r},可用的: {list(robot_obs.keys())}"
            )
        states = robot_obs[self.state_key]
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)
        states = states.to(self.device, dtype=torch.float32).reshape(self.num_envs, -1)

        return {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": list(self._task_descriptions),
        }

    @property
    def info_logging_keys(self) -> list[str]:
        # 让 RLinf 把官方判定和任务自报的成败都记进日志。两者长期不一致就说明
        # install_official_reward 的假设在这个任务上不成立,需要查。
        return ["official_success", "task_reported_success"]
