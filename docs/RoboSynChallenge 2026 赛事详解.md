# RoboSynChallenge 2026 赛事详解

> NeurIPS 2026 官方竞赛赛道 · Sim2Real 双臂机器人操作挑战赛
> 资料来源：小红书官方推广帖（深圳河套学院）+ 官网 robosyn-bench.net + 官方文档 edem-ai.github.io/RoboSynChallenge（截至 2026-07-30）

---

## 一、赛事总览

| 项目 | 内容 |
|---|---|
| 赛事全称 | RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills |
| 归属 | NeurIPS 2026 官方 Competition Track |
| 主办单位 | 深圳河套学院、跨维智能 |
| 赞助机构 | 跨维智能、腾讯云、松灵机器人 |
| 联合机构 | 香港中文大学（深圳）、香港中文大学、香港科技大学、复旦大学、中山大学、滑铁卢大学、Vector Institute 等 |
| 参赛对象 | 全球高校、科研院所、企业研发团队、独立研究者，个人或团队均可 |
| 团队规模 | 每队最多 5 人 |
| 总奖金 | 20,000 美元 |
| 报名时间 | 2026年7月13日起，阶段内全程开放 |
| 算力支持 | 官方提供"算力自由"支持（具体申请方式见报名页） |

赛事要解决的核心问题：真实机器人数据采集昂贵且难以覆盖所有场景变化，而仿真数据虽可规模化生成，却往往在 Sim2Real 迁移中失效。RoboSynChallenge 把这个问题变成一个**统一、可横向比较的评测基准**：所有队伍在同一套任务、同一套指标、同一台真机上被评测，成绩差异只反映数据方案、模型架构和训练方法本身的优劣。

---

## 二、赛制与关键时间线

比赛分两个阶段，全程约 5 个月：

| 阶段 | 时间 | 内容 |
|---|---|---|
| 报名开放 | 7月13日 | 开放报名，同步发布代码库、教程、数据与基线模型 |
| 训练与优化 | 7月13日 – 10月11日 | 队伍使用官方合成数据 + 少量真实示范训练策略，阶段内可持续报名 |
| 仿真初赛评测 | 10月11日 – 10月18日 | 官方在统一仿真环境中评测所有提交，不依赖真机 |
| 晋级公布 | 10月18日 | 公布晋级决赛的团队名单 |
| 决赛模型更新 | 10月18日 – 11月15日 | 晋级队伍继续迭代、更新模型 |
| 真机终评 | 11月15日起 | 组织方将最终策略部署到深圳河套学院具身智能与计算机视觉中心的统一双臂机器人平台，由官方运行评测（选手不接触真机） |
| 颁奖与展示 | 12月上旬（NeurIPS 2026 会期 12月6–12日前后） | 现场颁奖、成果展示、媒体报道 |

**两阶段的本质区别**：初赛是 *simulation-only*，全部队伍在仿真里比拼；决赛是 *real-robot only*，由赛事组织方在同一台真机上运行晋级团队提交的策略，从而排除个体硬件差异对结果的影响。

---

## 三、任务体系（10项官方双臂操作任务）

1短时序、目标清晰的基础操作，68

75234逐步过渡到需要双臂协调、精准控制和多阶段规划的复杂任务：

**入门级（Entry-level）**

- 桌面整理 Table Rearrangement
- 按铃 Click Bell
- 双臂倒水 Water Pouring
- 篮筐搬运 Handle Basket

**中等级（Mid-level）**
- 物品交接与放置 Items Hand-Over
- 打开抽屉并放置物品 Drawer Open-and-Place
- 操作搅拌器 Mixer Operating

**高等级（High-level）**
- 物品装配 Item Assembly
- 移液器操作 Manipulate Pipette
- 样品装载 Sample Loading

> 备注：官方代码库的任务名称枚举中还包含一个 `open_pan` 任务（开锅盖），推测是代码库中额外提供/预留的任务类型，宣传材料公布的正式评测任务为以上 10 项，具体是否计入最终评测以官方赛程通知为准。

官方网站为每个任务都提供了**仿真环境**和**真实环境**的对照演示视频，方便团队在训练前理解任务的具体形态、物体摆位和成功判定标准。

---

## 四、数据体系

### 4.1 真实数据（Real-World Data）

- 每个任务 **60 条** 真实双臂遥操作（teleoperation）采集的轨迹
- 采集平台由双臂机器人、顶部相机、灯光系统、遥操作系统和可控背景共同组成，统一硬件配置以减少不同实验室平台带来的干扰
- 覆盖 **5 种采集条件**：白底固定光照无干扰物 / 白底增强光照无干扰物 / 白底固定光照+2-3个干扰物 / 蓝底固定光照无干扰物 / 黄底固定光照无干扰物
- 覆盖 **4 种物体位置变化** 和 **3 种朝向设置**
- 数据格式为 LeRobot（支持 2.1 / 3.0 两种版本，提供互相转换脚本）

### 4.2 仿真数据（Synthetic Data）

- 每个任务 **1,000 条** 程序化生成的仿真轨迹
- 轨迹为多模态数据：RGB-D 观测、机器人状态、动作、接触事件、任务标注等
- 领域随机化（Domain Randomization）覆盖五大类：

| 类别 | 随机化内容 |
|---|---|
| 光照与场景外观 | 光源强度、位置、颜色；背景板颜色变化；桌面颜色和材质随机概率 |
| 物体几何与位姿 | 初始位置偏移、初始旋转范围、物体尺寸缩放与表面材质颜色 |
| 相机标定与视角 | 相机内参 fx/fy、相机位置 XYZ 偏移、相机横滚/俯仰/偏航扰动 |
| 机器人初始化 | 关节配置随机化、末端执行器位置变化、面向恢复行为的多样化可行起始状态 |
| 工作空间与干扰物 | 桌面高度上下浮动约 ±4cm；加入碗、杯子、玩具等无关干扰物增加场景复杂度 |

### 4.3 数据获取与生成工具

- 数据集托管在 Hugging Face（`RoboSynChallenge/cobotmagic_Real_{task_name}` 与 `RoboSynChallenge/cobotmagic_Sim_{task_name}`），可用 `huggingface-cli download` 直接下载
- 官方开源了 **EmbodiChain** 仿真引擎及配套的 RoboSynChallenge 扩展包，队伍可自行采集/扩展数据，而不必只依赖官方发布的固定数据集
- 仿真环境基于 EmbodiChain（v0.2.3），支持 Docker（预装 CUDA 12.8 / Vulkan / Python 3.11）和本地 `uv` 虚拟环境两种安装方式
- 每个任务的动作逻辑由 `action_config.json`（任务专属工作流图）与 `action_bank.py`（通用底层算子库）配合定义，两者通过共享缓存 `env.affordance_datas` 交换数据，理论上支持团队自定义/扩展新任务
- 若需多任务、仿真+真实混合训练，可用 `lerobot-edit-dataset` 工具或 `launch/collect_combined_dataset.sh` 脚本合并数据集
- 采集硬件建议：RTX 5060 Ti 级别 GPU 每任务采集建议控制在约 500 episode 以内以保证稳定性；RTX 4090 级别及以上不受此限制

---

## 五、评测方法

### 5.1 三大核心指标

| 指标 | 英文 | 含义 |
|---|---|---|
| 任务成功率 | Success Rate | 机器人能否稳定完成指定任务 |
| 动作步数 | Action Steps | 完成任务所需动作是否高效 |
| 推理时间 | Inference Time | 策略在真实执行中的响应效率 |

三项指标在**初赛和决赛中口径完全一致**，便于跨阶段纵向比较。官方强调"偶尔成功一次并不够"——好的方案需要同时兼顾稳定性、执行效率和部署速度，而不是只刷单次成功率。

### 5.2 两阶段评测流程

- **初赛（仿真）**：全部提交在 RoboSynChallenge 仿真环境中运行，不涉及任何真实机器人；官方统一执行评测并计算上述三项指标，据此筛选晋级团队。
- **决赛（真机）**：只在标准化真机平台上进行；晋级团队提交策略（模型 checkpoint + 部署代码），由赛事组织方在深圳河套学院具身智能与计算机视觉中心的统一双臂机器人平台上运行，选手不能亲自操作真机；同样计算三项指标，最终排名以真机结果为准并发布到官方排行榜（Leaderboard）。

### 5.3 策略提交接口（Deploy Your Policy）

参赛团队需要按官方接口规范封装自己的策略，核心是实现三个文件：

- `deploy_policy.py`：需实现 `get_model()`（加载模型）、`encode_obs()` / `encode_action()`（观测/动作格式转换，可选）、`eval()`（单步推理与环境交互主循环，必需）、`reset_model()`（每个评测 episode 开始前重置模型状态，可选但推荐）
- `deploy_policy.yml`：定义模型相关参数（checkpoint 路径、模型类型等）以及基础实验配置（`max_episodes`、`max_steps`、`seed`、`pytorch_device` 等），整份 YAML 会作为 `usr_args` 传入 `get_model()`
- `eval.sh`：命令行脚本，用于覆盖 `deploy_policy.yml` 中的默认参数并启动评测

`task_name` 需为以下枚举之一：`click_bell`、`handle_basket`、`water_pouring`、`table_rearrangement`、`items_handover`、`drawer_open_place`、`mixer_operating`、`item_assembly`、`manipulate_pipette`、`sample_loading`（以及代码库中额外提供的 `open_pan`）。这套统一接口保证不同团队的策略在同一套评测流程下被公平地比较。

---

## 六、官方基线策略（Baselines）

官方在 `policy/` 目录下提供了多个基线策略实现，方便队伍直接上手或对比：

| 基线 | 状态 | 说明 |
|---|---|---|
| ACT | 完整可用 | 基于 LeRobot 的 ACTConfig，用 `uv` 管理依赖；训练参数（batch size、chunk size、n-action-steps 等）均可通过命令行配置，支持多卡 DDP、AMP 混合精度、Weights & Biases 日志 |
| DP（Diffusion Policy） | 提供 | 与 ACT 类似的训练/评测流程 |
| PI0 | 完整可用 | 依赖 openpi 框架；区分 LoRA 微调（约需 46GB+ 显存，如 A6000）和全量微调（约需 100GB+ 显存，如 2×A100/H100）；提供 `finetune.sh` / `eval.sh` 脚本和归一化统计计算工具 |
| PI0.5 | 提供 | 训练流程与 PI0 类似 |
| Motus | 文档标注为 [TODO] | 环境搭建、数据准备、微调脚本等细节尚未完整发布，评测输出路径已定义（`eval_result/{task_name}/motus/...`） |

数据格式统一采用 LeRobot（2.1 / 3.0 两版本互转），采集到的仿真数据默认是 LeRobot 3.0，训练前可能需要用官方脚本 `convert_lerobot3.0_to_2.1.py` 转换。

---

## 七、奖励设置

| 奖项 | 金额 | 名额 |
|---|---|---|
| 一等奖 | 4,000 美元 | 1 队 |
| 二等奖 | 2,000 美元 | 5 队 |
| 三等奖 | 1,000 美元 | 6 队 |
| 合计 | 20,000 美元 | 12 队 |

除现金奖励外，优胜团队还将获得：获奖证书、登上 NeurIPS 2026 现场展示舞台的机会、官方媒体报道，以及国际机器人与 AI 社区的集中曝光。

---

## 八、关键链接

- 赛事主页：https://robosyn-bench.net/
- 报名链接：https://robosyn-bench.net/#/register
- GitHub 代码库：https://github.com/EDEM-AI/RoboSynChallenge
- 仿真引擎 EmbodiChain：https://github.com/DexForce/EmbodiChain
- 数据集（Hugging Face）：https://huggingface.co/RoboSynChallenge/datasets
- 入门教程文档：https://edem-ai.github.io/RoboSynChallenge/html/getting_started/overview.html
- 策略部署教程：https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
- 竞赛报告 PDF：见官网首页 "Read report" 按钮

---

*本文档内容综合自赛事推广帖、官方网站与官方技术文档，时间线、奖金等信息以赛事官网最新公告为准。*# RoboSynChallenge 2026 赛事详解

> NeurIPS 2026 官方竞赛赛道 · Sim2Real 双臂机器人操作挑战赛
> 资料来源：小红书官方推广帖（深圳河套学院）+ 官网 robosyn-bench.net + 官方文档 edem-ai.github.io/RoboSynChallenge（截至 2026-07-30）

