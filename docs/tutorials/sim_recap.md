# sim-RECAP 操作手册(人手版,不依赖 AI)

把 π\*0.6 的 **RECAP** 思想(价值函数 → advantage → 优势条件化策略,实现取自 Evo-RL,
收编在 `third_party/evo_rl/`)接入 RoboSynChallenge:真机上需要人打成败标签、人接管纠正,
仿真评估器本身就是免费的标注器,所以整个闭环无人值守。每一步都有现成脚本(`launch/recap/`),
按编号顺序跑即可;机器相关路径集中在 `launch/recap/_common.sh`。

## 思想

RL 被拆成三个监督学习问题,全程离线、不改模型结构:

1. **成败标签**:rollout 时评估器自动判 success/failure(专家数据天然全 success);
2. **价值函数**(pistar06,SigLIP + Gemma-3-270m 全量微调):目标 = 归一化负剩余步数(失败罚 `c_fail`),
   同时编码"能不能成"和"还要多久";
3. **ACP 优势条件化**:n-step advantage 在任务内按 **top-30%** 二值化(Evo-RL 原实现,不做后处理),
   训练时把文本 `Advantage: positive/negative` 拼进 prompt(30% dropout),**部署时永远挂 positive**。
   失败数据不浪费,每一段做对的部分都被回收成正样本。

数据池必须同时含**成功与失败的 rollout** + 专家数据,否则价值函数学的是"辨认动作风格"而不是成败。

## 一轮流程(以 sample_loading round1 为例)

```bash
# 0 清场看卡:别误杀别人的 covered_eval / collect_until_valid
nvidia-smi; bash launch/recap/stop.sh <cmdline子串>            # 需要时用

# 1 rollout(2 分片并行,自动成败标签)→ lerobot_dataset/rollouts/sample_loading_round1(v2.1)
bash launch/recap/01_rollout.sh sample_loading round1 pi05_sample_loading sample_loading 28000 150

# 1b 人工三视角复核视频,改标签(三处同步)
bash launch/recap/02_set_label.sh sample_loading round1 91 failure

# 2-4 数据池:专家(清洗版)前 200 集 + rollout 150 集,写 episode_success
bash launch/recap/03_build_pool.sh sample_loading round1 \
    /home/phl/FermiBotNas/dataset/RoboSynChallenge/Sim_clean_filtered/cobotmagic_Sim_sample_loading 200

# 5 价值训练(脱离会话;~12h 训满,但通常 3000 步就够)
bash launch/recap/04_value_train.sh sample_loading round1

# 6a 质检选档(60 集子集,每档 ~10 分钟)→ 选 advantage 信号最强且分离度 ≥0.3 的档
bash launch/recap/05_value_qc.sh sample_loading round1 001500 002000 003000   # 只列已存在的档
bash launch/recap/stop.sh lerobot_value_train                     # 发布前必须停训

# 6b-7 全量推理写回 + 发布 no_reward / reward 两版 v2.1 到 NAS 与本地(~1.5h)
setsid nohup bash launch/recap/06_publish.sh sample_loading round1 003000 > /tmp/pub_round1.log 2>&1 < /dev/null & disown

# 8 ACP 微调(对照组加 SIMRECAP_INDICATOR_KEY=none)
bash launch/recap/07_acp_finetune.sh sample_loading round1 sample_loading_round1 0 \
    ./checkpoints/pi05_base_robosynchallenge_full/sample_loading/28000/params

# 9 官方 random 协议评估(自动取最新 checkpoint,自动挂 Advantage: positive)
bash launch/recap/08_eval.sh sample_loading round1 sample_loading_round1 100
# 对照组评估必须同样设 none,否则 prompt 会被追加标签:
SIMRECAP_INDICATOR_KEY=none bash launch/recap/08_eval.sh sample_loading round1 sample_loading_round1_sft 100
```

下一轮:`01_rollout.sh sample_loading round2 pi05_sim_recap sample_loading_round1 19999 150`(脚本会从 checkpoint 的
assets 推出 `SIMRECAP_REPO_ID`),`03_build_pool.sh sample_loading round2 lerobot_dataset/simrecap_sample_loading_round1`
(上轮 reward 池:自带边车,失败集标签会被正确恢复;脚本会先去掉它的 `complementary_info.*` 三列再与新 rollout 合并),
`07` 的权重指向 `checkpoints/pi05_sim_recap/sample_loading_round1/19999/params`。

## 每步的判定规则

| 步骤 | 看什么 | 合格线 / 处理 |
|---|---|---|
| 01 rollout | 成功率、`validate_lerobot_dataset` 四门 | `random_rollout` 下 sample_loading ≈7%;全败也可继续(负样本) |
| 02 复核 | 三视角终态:管子是否留在架孔内直立 | 评估器在稳定计数触发后就结束,可能漏掉后来掉出 |
| 03 数据池 | 专家:rollout 比例 | round1 用 200:150;专家用 NAS `Sim_clean_filtered`(756 集),不用本地 1000 集原始版 |
| 04 训练 | wandb loss | 参考:5.3→2.1(500)→1.4(2000)→1.2(3000)→0.82(5000)→0.73(6000+ 平台) |
| 05 质检 | 成功−失败首帧 value 差;advantage std;近零占比 | 差 ≥0.3 的档里取 std 最大者。round1:1500 步差 0.11(欠训);**3000 步差 0.43、std 0.030(选)**;6500 步差 0.69 但 std 0.005、96% 帧≈0(记忆化,弃) |
| 06 发布 | 日志 `ACP stats`、"三列存活确认"、NAS 就绪行 | 专家帧 indicator≈10% 为正是 top-30% 混算的正常结果 |
| 08 评估 | 同协议对比 ACP vs 纯 SFT(no_reward 或 `SIMRECAP_INDICATOR_KEY=none`) | 提升归因需要这组对照 |

## 数据版本与位置

对外一律 **v2.1**;v3.0 只存在于隐藏工作目录 `lerobot_dataset/.simrecap_work/<task>_<tag>/`
(价值栈只认 v3.0)。发布产物:

| 位置 | 内容 |
|---|---|
| `lerobot_dataset/rollouts/<task>_<tag>/` | rollout(v2.1 + `episode_success.json`) |
| NAS `recap_no_reward_dataset/simrecap_<task>_<tag>/` | 合并池,未过价值模型(纯 SFT 对照) |
| NAS `recap_reward_dataset/simrecap_<task>_<tag>/` + 本地 `lerobot_dataset/simrecap_<task>_<tag>/` | 合并池 + `value/advantage/acp_indicator_<tag>` 三列(ACP 训练用,已链进 `policy/pi05/training_data/RoboSynChallenge/`) |

标签载体:集级成败只认 **`episode_success.json` 边车**(打过标的合并池转成 v2.1 后 jsonl 里也会带该字段,但转换/打标脚本不读它;rollout 交付版只有边车);
两个方向的格式转换器都会丢边车,脚本负责带回;上轮池当专家用而缺边车时 `03` 会拒绝。
三列 advantage 标签是普通帧级数据列,v2.1 原样携带,训练时由 openpi 的 `ACPAdvantageTag` 现场拼成 prompt 文本。

## 环境(三套,脚本已各自选对,列出以便排障)

| 用途 | 解释器 |
|---|---|
| rollout / 评估 / v2.1 训练读取校验 | `policy/pi05/.venv/bin/python` |
| 转换、打标、剥元数据、边车 | `~/miniconda3/envs/robosyn/bin/python`(lerobot 0.4.4 + pyarrow,pandas 3) |
| 合并、价值训练、价值推理 | `~/miniconda3/envs/evo-rl/bin/python` + `PYTHONPATH=third_party/evo_rl/src`(pandas 2) |

## 常见故障(全部踩过)

- **杀进程把自己杀了**:`pgrep -f` 会匹配当前 shell;用 `launch/recap/stop.sh`(排除自身),或 `'xx[x]'` 括号技巧。
- **显存被幽灵占用**:杀 eval/采集主进程后 image-writer 子进程存活;`stop.sh` 会列出仍持卡的 PID,逐个清。
- **merge 报 HF Hub 404 / 找不到 info.json**:`--root` 传法错误;脚本已用 `HF_LEROBOT_HOME`。`collect_parallel_validated.sh:109` 是错误示范,别照抄。
- **merge/convert 在 pandas 上崩**:parquet 带 HF 扩展 dtype;脚本会先剥元数据(`_common.sh strip_meta`)。
- **训练被会话重启杀掉**:脚本用 `setsid nohup` 脱离;自己起长任务也要这样。
- **wandb 404**:机器 `~/.netrc` 账号与浏览器账号不一致;`wandb login --relogin <key>` 后重启训练。当前登录 `puheliang`。
- **价值训练目录已存在**:`FileExistsError`(训练器要求 output_dir 不存在,启动脚本/日志放在旁边的 `.launch` 目录);删除或 `--resume=true --config_path=.../checkpoints/last/pretrained_model/value_train_config.json`。
- **发布被拒绝**:价值训练还在跑(会读 merged_v30)/ 同 tag 重复发布(列已存在)/ merged_v30 没打标——按提示处理。
- **评估找不到 checkpoint**:20k 步训练只存 `10000/19999`,`08_eval.sh` 自动选最新;手动跑 eval.sh 要加 `--checkpoint_id`。
- **norm stats 跨轮不重算**:`finetune.sh` 只看配置名目录;`07_acp_finetune.sh` 按 repo_id 路径检查。
- **试管贴架子无解**:采集用 `configs/<task>/random_rollout/`(几何推导见其 README),评测仍用官方 `random`。

## 参考

- Evo-RL 收编说明:`third_party/evo_rl/VENDORED.md`;打标语义:任务内全帧混算 top-30%,`force_intervention_positive` 在仿真数据上是 no-op。
- π\*0.6 论文(arXiv 2511.14759):同样的分位规则,迭代阶段比例 ~40%;人工纠正帧强制为正(仿真无此项)。
- 组件代码:`scripts/eval_policy.py --rollout_save`、`scripts/label_rollout_dataset.py`、`policy/pi05/src/openpi/policies/libero_policy.py` 的 `ACPAdvantageTag`、`policy/pi05/src/openpi/training/config.py` 的 `pi05_sim_recap`(环境变量驱动)。
