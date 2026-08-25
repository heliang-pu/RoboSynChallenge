# 踩过的坑(每条都发生过)

1. **杀进程把自己杀了**:`pkill -f "xxx"` / `pgrep -f "xxx"` 会匹配当前 shell 自己的命令文本
   (它含同样字符串)。用 `pgrep -f 'xx[x]'` 括号技巧,或先把 PID 写到文件再按号 `kill -9`。
2. **孤儿子进程拖住显存**:`kill -9` eval/采集主进程后,记录器 spawn 的 image-writer 子进程
   (cmdline 与父进程相同)存活,`nvidia-smi` 仍记账 20+GB,进程列表却查不到父进程。
   清理:`nvidia-smi --query-compute-apps=pid` 列出持卡 PID,逐个看 `/proc/<pid>/cmdline` 后杀。
3. **`lerobot_edit_dataset merge` 的 `--root` 是错的**:`LeRobotDataset(repo_id, root=)` 的 root 是
   数据集目录本身,传 `--root` 会把所有输入指向同一目录 → 找不到 info.json 后回落到 HF Hub 404。
   正确:`HF_LEROBOT_HOME=<父目录>`,不传 `--root`,repo_id 用相对路径(可含一个 `/`)。
   仓库自带的 `launch/collect_parallel_validated.sh:109` 用的正是这种错误传法,不要照抄。
4. **parquet 扩展元数据炸 pandas**:记录器/上转器写的 parquet 带 HF datasets 扩展 dtype
   (`cube_pose`/`rack_pose` 4×4 矩阵列)。merge 的 pandas 路径在 pandas 3(robosyn 环境)读
   `tasks.parquet` 崩、在 pandas 2(evo-rl)写 `rack_pose` 崩。修法:合并/转换前用不 import
   datasets 的裸 pyarrow 进程重写 `meta/**` 与 `data/**` parquet,丢掉 schema 级和字段级元数据
   (数值不变,幂等)。`launch/recap/_common.sh` 的 `strip_meta`(01/03/06 均调用)与 `run_sim_recap_round.sh` 阶段 3 已内置。
5. **两个方向的转换器都丢非标准文件**:`convert_lerobot3.0_to_2.1.py` 重建目录并留 `<name>_v3.0`
   备份;`convert_dataset_v21_to_v30` 只重建 data/meta/videos,其余文件留在 `<name>_old/`。两者都会让
   `episode_success.json` 消失——转完必须把边车复制回去(`_common.sh` 的 `to_v21`、`03_build_pool.sh`
   已内置)。**上轮合并池当专家用时若边车丢失,失败集会被整体标成 success**,脚本现在会拒绝这种输入。
   另:上转器 `--push-to-hub` 默认 True,必须显式 `--push-to-hub false`。
6. **会话重启杀训练**:Bash 后台任务是会话子进程,Claude Code 重启即 SIGHUP。12 小时的训练必须
   `setsid nohup … < /dev/null & disown`(验证:`ps -o sid,ppid` 应为自身 SID、父进程 1)。
   重启事故:死在 step 440,差 60 步没到第一个存档,50 分钟白跑。
7. **wandb 账号不一致 → 404**:机器 `~/.netrc` 的 key 决定 run 归属;浏览器账号不同就 404。
   查:`wandb.Api().viewer.entity`。切换:`wandb login --relogin <key>` 后必须重启训练。
8. **价值函数记忆化**:小池子上 loss 持续下降不等于更好——V 背下每条轨迹回报后 n-step advantage
   按 Bellman 恒等式归零(6500 步时 96% 帧 |A|<0.01)。永远用 `launch/recap/05_value_qc.sh` 比较多档 checkpoint。
9. **质检别做全量**:16 万帧 × 多档 checkpoint 与训练共享 GPU 时每档 2 小时;用 `--dataset.episodes`
   子集(60 集:10 成 + 30 败 + 20 专家)每档 10 分钟。质检在数据集**副本**上做,别写训练中的原集。
10. **试管贴架子的无解场景**:官方 `random` 里试管 x∈[0.45,0.68] y∈[-0.28,0]、架子 x∈[0.63,0.70]
    y∈[0,0.15],x 区间重叠、y 在 0 相接,最坏情况直接接触;架子 27.5×11.6cm(含 1.1 缩放)旋转外沿
    14.9cm。采集设置:`random_rollout`(roll ±20°)试管 y ≤ −(0.149+0.008+余量 0.03)≈ −0.19;
    `syn_tube_tilt`(±35°)y ≤ −(0.149+0.065+0.02)≈ −0.235。依据见各配置目录 README。评测协议不改。
11. **评估器判成功但视频看是失败**:稳定计数触发即 break,之后试管掉出评估器看不到。人工三视角
    复核后改边车(round1:ep91 成功→失败,10/150)。
12. **专家数据版本**:`/home/phl/workspace/dataset/cobotmagic_Sim_sample_loading` 是 1000 集原始版;
    NAS `Sim_clean_filtered` 是 756 集清洗版。用清洗版。
13. **pi05 `eval.sh` 的 XLA 显存比例**:已改为可被环境变量覆盖(`XLA_PYTHON_CLIENT_MEM_FRACTION`),
    并行分片各 0.32;单跑默认 0.4(≈20GB),启动前确认空闲显存,否则 JAX 直接 OOM。
14. **GPU 上常有别人的活**:用户会同时跑 `/home/phl/Datacollect_T/covered_eval_run/`(~22GB)和
    `collect_until_valid` 专家采集(三层监督链,杀 worker 会被自动重启,要杀 `collect_until_valid.sh`
    整树)。动手前 `nvidia-smi` 看清,别误杀,也别在满卡时起新任务。
15. **6 进程争用让推理慢 10 倍**(17s/次 vs 1.7s/次)。大批量 rollout 前清场,双分片并行可 ~2h 录 150 集。
16. **不存在的 delta 掩码 bug**:曾误判 `LeRobotEmbodiChainDataConfig` 的 delta 掩码只覆盖 7 维——
    实际 340 行的 `(6,-1)` 属于 LIBERO 配置,EmbodiChain 用的是正确的 `(6,-1,6,-1)`。
    审代码时把每个匹配归属到所在类,用真实配置管线验证,不要手搓前提再当证据。
17. **发布会原地改写 merged_v30**:阶段 6 推理把三列写进 `merged_v30/data/*.parquet` 并改 info.json。
    价值训练还在读它时发布 = 数据竞争;`06_publish.sh` 会拒绝(检测 `lerobot_value_train` 进程)。
    同一 tag 重复发布会叠列,也会被拒绝——换 tag 或重建 merged_v30。
18. **norm stats 跨轮误判**:`finetune.sh` 只看 `assets/<config>/` 目录是否存在,而 stats 实际按
    `assets/<config>/<repo_id>/` 存;同一配置名第二轮不会重算 → openpi 静默跳过归一化。
    `07_acp_finetune.sh` 按 repo_id 路径检查并计算。
19. **质检的 ind:* 列是子集内 top-30%**,与全量阈值不同,只看趋势;全量比例看 `06` 日志的 `ACP stats`。
20. **`value_train` 输出目录已存在会直接报错**(`FileExistsError`),续训需
    `--resume=true --config_path=<out>/checkpoints/last/pretrained_model/value_train_config.json`。
21. **两个训练器都会"吃掉"输出目录**:`lerobot_value_train` 要求 output_dir 不存在(FileExistsError);
    openpi `train.py` 会清空重建 checkpoint 目录(`--overwrite` 时连已有 checkpoint 一起删)。启动脚本和日志
    放在旁边的 `<out>.launch/`,04/07 已如此;不要往输出目录里预放文件。
22. **上轮 reward 池当专家用时列不匹配**:它带 `complementary_info.*` 三列而新 rollout 没有,lerobot merge
    校验 features 一致会直接失败。`03_build_pool.sh` 会先用 `remove_feature` 去列(缓存 `<cache>_noci`)。
23. **pi05_sim_recap 的策略做 rollout/评估必须带对 `SIMRECAP_REPO_ID`**:norm stats 按 repo_id 存在
    checkpoint 的 assets 里,不一致会 FileNotFoundError;`01_rollout.sh` 从 checkpoint assets 自动推导,
    `08_eval.sh` 按 tag 推导;对照组模型评估要 `SIMRECAP_INDICATOR_KEY=none`。
24. **价值 checkpoint 把模型绝对路径写死在 config.json 里**(`vision_repo_id`/`language_repo_id` =
    训练机上的 `/home/<user>/workspace/models/google/...`)。拷到另一台机器推理时先
    `grep -rl '/home/phl/' <ckpt>/ --include='*.json' | xargs sed -i 's#/home/phl/#/home/<user>/#g'`,
    否则 transformers 把路径当 HF repo id 报 `HFValidationError`。远端 evo-rl 环境还需手动
    `pip install transformers==4.53.3 scipy`(Evo-RL 的依赖表没带)。
25. **strip_meta 会丢任务指令文本**:v3.0 的 `meta/tasks.parquet` 把任务字符串存在 pandas 索引
    (`__index_level_0__`)里;strip_meta 重建表时丢掉"索引列"元数据,转 v2.1 后 `tasks.jsonl`
    的 task 退化成 `0`。因 `prompt_from_task=True`,训练 prompt 变成 "0" 而评估用真实指令 →
    训练/评估不一致。strip_meta 现在跳过 `tasks.parquet`。已交付数据集可直接重写 tasks.jsonl
    (单任务时指令已知)。合并前剥分片元数据也会经此丢失,同样已修。
