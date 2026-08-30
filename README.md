<div align="center">
<h1>RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills</h1>

<h2 align="center"> 👉<a href="https://edem-ai.github.io/robosynchallenge.github.io/">Webpage</a> | <a href="https://edem-ai.github.io/RoboSynChallenge/html/">Document</a> | <a href="">Paper</a> | <a href="https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard">Leaderboard</a></h2>

![image](misc/robosynchallenge-pipeline.png)

</div>

---

<!-- branch-readme:begin -->

> **分支导航** — 本仓库按主题分支开发，每个分支的说明就在各自 README 的这个位置。
>
> [`main`](../../tree/main) 基线 · [`sim-recap`](../../tree/sim-recap) RECAP 价值函数 · [`feat/rtc-async-pi05`](../../tree/feat/rtc-async-pi05) 实时分块与异步执行 · [`feat/lila-wam`](../../tree/feat/lila-wam) LiLa-WAM 与覆盖度采集 · **`feat/realtime-vla-pi05`（当前）** 推理加速 · [`ppo-post-training`](../../tree/ppo-post-training) PPO 后训练

## 本分支：`feat/realtime-vla-pi05` — pi0.5 推理加速

用 [`dexmal/realtime-vla`](https://github.com/dexmal/realtime-vla) 的 Triton kernel 加速 pi0.5 推理，OpenPI checkpoint 保持只读。

RTX 4090 上以 `pi05_click_bell_baseline/19999` 实测，端到端 **80.89 ms → 43.26 ms（1.87×，延迟降 46.5%）**，完整数据见 `policy/pi05/realtime_vla/RESULTS.md`。

- `convert_checkpoint.py`：JAX checkpoint → 加速器权重
- `tokenizer_adapter.py`：把 OpenPI 原生的 paligemma tokenizer 桥接成 realtime-vla 期望的 HF tokenizer 形状
- `accelerated_policy.py`：加速推理路径
- `benchmark.py` / `benchmark_e2e.py` / `benchmark_jax.py`、`validate_outputs.py`：分层基准与输出一致性校验

用法见 `policy/pi05/realtime_vla/README.md`。

---

<!-- branch-readme:end -->

# Contents

- [Contents](#contents)
- [Installtion](#installtion)
- [Datasets](#datasets)
- [Training and Evaluation](#training-and-evaluation)
- [Released Checkpoint Results](#released-checkpoint-results)
- [LeaderBoard](#leaderboard)

# Installtion
Based on the [**EmbodiChain**](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html), we offer both `docker` and `local` installation methods. For detailed installation instructions, please refer to [**Installation Document**](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html).

# Datasets
We provide 1,000 pre-collected trajectories per task as part of the open-source release **RoboSynChallenge** Dataset. The datasets hosted on HuggingFace are available at [here](https://edem-ai.github.io/robosynchallenge.github.io/#/data).

However, we still strongly recommend users to perform data collection themselves. For detailed data collection instructions, please refer to [**Data Collection Document**](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/collect_data.html).


# Training and Evaluation
Currently, RoboSynChallenge integrates training and evaluation for <a href="https://github.com/Physical-Intelligence/openpi">PI0</a>, <a href="https://github.com/Physical-Intelligence/openpi">PI0.5</a>, and <a href="https://github.com/thu-ml/Motus">Motus</a>. Detailed procedures can be found in the documentation for the corresponding strategies: 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/index.html">Webpage</a>.
In addition, you can easily extend your own policys for training and evaluation by following the documentation 👉<a href="https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html.html">Webpage</a>.

# Released Checkpoint Results

100-episode simulation evaluations for the released ACT and Diffusion Policy checkpoints are published in [`evaluation_results`](evaluation_results/README.md). The machine-readable result file pins success rate, action steps, millisecond inference time, Hugging Face checkpoint revisions, and the `random` protocol configuration.

# LeaderBoard
The full leaderboard and setting can be found in: https://edem-ai.github.io/robosynchallenge.github.io/#/leaderboard.
