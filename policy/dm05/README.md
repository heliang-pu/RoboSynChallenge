# DM05 (OpenDM) Policy for RoboSynChallenge

将原力灵机(Dexmal)开源的 [OpenDM / DM0.5](https://github.com/dexmal/opendm) VLA 模型接入
RoboSynChallenge 统一评估接口。

## 架构

DM0.5 以独立 HTTP 服务方式推理(`POST /v1/infer`),本适配器
([deploy_policy.py](deploy_policy.py) / [dm_model.py](dm_model.py))只做:

1. 观测编码:`cam_high` → Head、`cam_left_wrist` → Left wrist、`cam_right_wrist` → Right wrist,加上 `robot/qpos` 状态;
2. HTTP 调用,取回绝对关节位置 action chunk;
3. 逐步执行前 `dm_step` 个动作。

模型服务与仿真评估跑在**两个独立环境**里,仿真环境不需要装 opendm 的任何依赖。

## 1. 准备 opendm 环境(独立 conda 环境)

```bash
# opendm 仓库已 clone 在 policy/dm05/opendm
conda create -n opendm python=3.10 -y
conda activate opendm
cd policy/dm05/opendm
pip install -e .
# 可选:低延迟 fast 后端(需要 TensorRT / Triton / torch>=2.5)
pip install -e ".[fast-infer]"
```

## 2. 下载 checkpoint

```bash
# 基座模型(3 图 + 14 维状态/动作,零样本可直接试)
hf download Dexmal/DM05 --local-dir policy/dm05/opendm/checkpoints/DM05
```

自己在挑战赛数据上 SFT 的 checkpoint 直接用训练输出目录即可
(注意目录里要带训练时的 `norm_stats.json`)。

## 3. 启动推理服务

```bash
conda activate opendm
bash launch/run_dm05_server.sh policy/dm05/opendm/checkpoints/DM05
# 自训 SFT checkpoint 示例:
bash launch/run_dm05_server.sh /path/to/ckpt --exp playground/dm05_sft_demo.py --dataset-name <your_dataset>
```

服务默认监听 `http://127.0.0.1:7891`,`--port` / `--backend fast` 等参数见 `-h`。

## 4. 运行评估(仿真环境)

```bash
python scripts/eval_policy.py --config policy/dm05/deploy_policy.yml \
    --task_name item_assembly --setting clear
```

## 配置要点([deploy_policy.yml](deploy_policy.yml))

| 字段 | 说明 |
| --- | --- |
| `server_url` | 推理服务地址,跨机评估时改为远端 IP |
| `dm_step` | 每次推理执行的动作步数,不要超过服务端 `chunk_size` |
| `robot_type` | norm_stats 的机器人 profile。零样本基座模型用 `Aloha`;自训 SFT 通常留 `null` |
| `control_mode` / `speed` | 文本条件字段。基座模型需显式给出;SFT 数据没有就留 `null` |
| `state_indices` | qpos 维度选择/重排,用于把仿真 qpos 对齐到 checkpoint 的状态维度 |

**维度对齐提醒**:`observation.state` 的长度和顺序必须与 checkpoint 的
`norm_stats.json` 完全一致。本挑战 Double-Piper 的 qpos 是 16 维
(双臂 6+6 关节 + 每臂 2 个夹爪指关节),而零样本基座 DM05 的 Aloha profile 是
14 维,零样本评估时需通过 `state_indices` 挑出 14 维(每臂只保留一个夹爪维度),
且关节顺序/夹爪取值范围与 Aloha 约定未必一致——零样本结果仅供参考,
**推荐路径是用本仓库生成的 LeRobot 数据按 [opendm 的数据注册与 SFT 流程](opendm/docs/en/dm05_finetuning.md)
微调后再评估**,此时状态/动作约定天然一致,`robot_type`/`state_indices` 都不用设。

## 已知差异 / 注意事项

- 动作语义为绝对关节位置(与本项目 eval 接口一致);若 SFT 时用了
  `relative` action mode,需在服务端保持一致并自行处理,当前适配器只支持绝对 qpos。
- 服务按 batch size 1 串行处理请求,评估并行度 `num_envs=1`。
- 首次用 `--backend fast` 启动会先导出 ONNX、构建 TensorRT engine,耗时较长属正常。
