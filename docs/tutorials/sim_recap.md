# sim-RECAP:仿真里的优势条件化迭代训练(无需人在回路)

把 π\*0.6 的 **RECAP** 思想(价值函数 → advantage → 优势条件化策略,实现取自
[Evo-RL](https://github.com/MINT-SJTU/Evo-RL),已收编在 `third_party/evo_rl/`)
接入 RoboSynChallenge:真机上需要人打成败标签、人接管纠正,而仿真评估器本身
就是免费无噪声的标注器,因此整个闭环可以完全无人值守。

## 思想

RL 被拆成三个监督学习问题,全程离线、稳定,不改一行模型结构:

1. **成败标签**:rollout 时评估器自动判 success/failure(专家数据天然全 success);
2. **价值函数**(pistar06):SigLIP+LLM 骨干,分布式 bin 输出,目标 =
   归一化负剩余步数(失败罚 `c_fail`)——同时编码"能不能成"和"还要多久";
3. **ACP 优势条件化**:按任务内 top-30% 把 n-step advantage 二值化为 0/1,
   训练时把字面文本 `Advantage: positive/negative` 拼进 prompt(30% dropout),
   **部署时永远挂 positive**——失败数据不浪费,成为教会模型"什么是坏"的负样本。

关键设计:数据池必须同时含**成功与失败的 rollout** + 专家数据。若失败样本只来自
rollout、成功样本只来自专家,价值函数会学到"辨认动作风格"的捷径而不是任务成败。

## 数据版本约定:对外统一 v2.1

专家数据给 v2.1(v3.0 也接受),最终产物是 v2.1。价值函数栈(Evo-RL)只认
v3.0,所以 v3.0 只作为**内部中间产物**存在于隐藏工作目录
`lerobot_dataset/.simrecap_work/<task>_<tag>/`,阶段 7 转回 v2.1 发布,
对外目录里永远只有 v2.1。

## 一轮闭环(一条命令)

```bash
bash launch/run_sim_recap_round.sh click_bell random \
     pi05_base_robosynchallenge_full click_bell round1 200 0
```

七个阶段自动串联(`START_STAGE=N` 可断点续跑;下文 `work/` 指隐藏工作目录):

| 阶段 | 做什么 | 产物 |
|---|---|---|
| 1 | π_k 无头 rollout N 集,自动记录成败 | `work/rollout_v30/` + `episode_success.json` 边车 |
| 2 | 专家数据准备:v2.1 复制后上转 v3.0(按源数据指纹缓存,同一份数据只转一次,后续轮秒级软链),v3.0 直接软链 | `work/expert_v30/` |
| 3 | 专家 + rollout 合并(顺序固定:专家在前) | `work/merged_v30/` |
| 4 | 写 `episode_success` 列进 meta/episodes(前缀带边车时按边车,否则全 success) | 同上(带标签) |
| 5 | 训练 pistar06 价值函数 | `outputs/value_train/value_<task>_<tag>/` |
| 6 | 逐帧 value/advantage/indicator 写回数据集 | `complementary_info.acp_indicator_<tag>` 列 |
| 7 | 导出标签边车 → 转 v2.1(校验 indicator 存活)→ 发布 | `lerobot_dataset/simrecap_<task>_<tag>/`(v2.1,含边车),并链接进 pi05 训练目录 |

标签的跨轮传递:发布的 v2.1 数据池自带 `episode_success.json` 边车
(v2.1 的 meta 不保留自定义列)。下一轮把它当 `expert_dataset` 时,
阶段 4 自动改用这个边车恢复逐集标签——上轮池子里的失败集不会被错标成 success。

然后按脚本末尾提示做 ACP 微调(阶段 8):

```bash
# config.py 的 pi05_sim_recap 中确认 repo_id / acp_indicator_key / weight_loader
bash policy/pi05/finetune.sh pi05_sim_recap click_bell_round1 0
```

训完直接评估——推理链路会自动给 prompt 追加 `Advantage: positive`,不需要
改 deploy 配置。下一轮把 `round_tag` 递增、`weight_loader` 指向新 checkpoint、
`expert_dataset` 换成本轮合并池,数据池随迭代滚雪球。

## 单独采集 rollout(不跑完整闭环)

只想录一批带成败标签的 rollout 数据时,直接用 eval.sh:

```bash
cd policy/pi05
bash eval.sh sample_loading random_rollout pi05_sample_loading sample_loading 0 \
    --checkpoint_id 28000 --max_episodes 100 --headless True \
    --rollout_save True --rollout_save_path lerobot_dataset/my_rollout \
    --eval_video_log False
```

- 数据集落在 `<save_path>/<robot>_<scene>_<task>_NNN/`(记录器自动建子目录),
  `episode_success.json` 边车在数据集目录内
- 交付 v2.1:复制一份后用 `scripts/convert_lerobot3.0_to_2.1.py` 转换,
  **转换器会丢弃非标准文件,转完记得把边车复制回去**

## 采集设置与评测设置分离

评测必须用官方 `random`;采集可以用任务专属的 `random_<用途>` 设置排除
无解场景。已有示例:`configs/sample_loading/random_rollout/` 把试管出生范围
收窄到不会贴住架子(官方范围最坏情况两者直接接触,episode 无解,只产生
无信息量的失败样本;几何依据见该目录 README)。新任务照此模式复制官方
random 配置后微调即可,**不要改官方 random 本身**。

## 运维注意

- **中途停止 rollout**:记录器会 spawn image-writer 子进程,只杀主进程会留下
  孤儿进程拖住 20+GB CUDA 上下文不释放(表现为进程消失但 nvidia-smi 仍记账)。
  正确做法:`pgrep -af eval_policy` 找全 PID 一起 kill,再确认显存归零。
- **显存预算**:pi0.5 评估会按 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` 预分配约
  20GB(48G 卡),启动前确认空闲显存足够,否则 JAX 直接 OOM。
- **一集时长**:sample_loading 失败集跑满 600 步(任务注册的
  `max_episode_steps=600` 压过配置里的数值),GPU 无争用时约 4-7 分钟/集,
  估算大批量采集时长要按失败集为主计算。

## 各组件位置

| 组件 | 文件 |
|---|---|
| rollout 采集 + 成败边车 | `scripts/eval_policy.py` 的 `--rollout_save True`(利用 EmbodiChain 的 `save_failed_episodes`,失败集也落盘) |
| 标签写入(边车/常量/合并前缀三种模式,带硬校验门) | `scripts/label_rollout_dataset.py` |
| 价值训练 / advantage 写回 | `launch/run_value_train.sh` / `launch/run_value_infer.sh`(包装收编的 Evo-RL CLI) |
| ACP prompt 注入(训练带 dropout,推理自动 positive) | `policy/pi05/src/openpi/policies/libero_policy.py` 的 `ACPAdvantageTag` + `LeRobotEmbodiChainDataConfig.acp_indicator_key` |
| 示例训练配置 | `policy/pi05/src/openpi/training/config.py` 的 `pi05_sim_recap` |
| 价值函数实现(上游收编,勿改) | `third_party/evo_rl/`(来源与删减见其 VENDORED.md) |

## 环境

- 阶段 1/2/3/4/7:仿真环境(conda `robosyn` 或根目录 `.venv`,lerobot 0.4.4 + pyarrow)
- 阶段 5/6:Evo-RL 环境——`cd third_party/evo_rl && uv venv --python 3.10 && uv pip install -e .`,
  或复用已有 conda `evo-rl` 环境(脚本会用 `PYTHONPATH` 保证跑的是收编代码)
- 阶段 8(ACP 微调):pi0.5 的 uv 环境(不变)

## 经验参数

- `--acp.n_step 50`:与 pi0.5 的 action_horizon 对齐
- `--acp.positive_ratio 0.3`:任务内 top-30% 为 positive
- `acp_tag_dropout 0.3`:让模型同时学会带/不带标签两种条件
- 低成功率任务(如 sample_loading)首轮多掺专家数据,先用 SFT 把成功率拉到
  两位数再进循环,否则 positive 样本几乎全来自专家,迭代提升慢
