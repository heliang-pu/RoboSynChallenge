# 评估环境

评估代码在本仓库（`scripts/eval_policy.py`、`policy/*/eval.sh`、`policy/*/deploy_policy.py`），
但跑它们的虚拟环境**刻意放在仓库外**，与 `policy/*/.venv`（uv 按项目自建、仓库自带约定）区分开。

放在外面的三个理由：

1. `.venv` 在 `.gitignore` 里，`git clean -xfd` 专清 ignored 文件，跑一次会把几十 G 环境清光；
2. 提交比赛代码时打包仓库目录即可，不用挑不用排除；
3. 这些是**环境**不是代码，混进代码仓库会让「提交什么」变得含糊。

约定位置：`$WS_ROOT/venvs/`（`WS_ROOT` = 仓库的上一级，见 [CLAUDE.md](../CLAUDE.md) 的路径基准）。

## 四个环境

| venv | Python | 关键版本 | 用途 |
|---|---|---|---|
| `eval_venv` | 3.11 | **lerobot 0.1.0** | pi0.5 评估 |
| `eval_venv_act` | 3.11 | **lerobot 0.3.3** | ACT 评估 |
| `venv-train` | 3.11 | torch 2.11+cu128 | 训练 |
| `smolvla_venv` | 3.12 | lerobot 0.6.2（editable） | SmolVLA |

### `eval_venv` 与 `eval_venv_act` 不能混用

lerobot 在 0.1.0 → 0.3.3 之间改了模块布局：

| | 模块路径 |
|---|---|
| 0.1.0 | `lerobot.common.policies.act` |
| 0.3.3 | `lerobot.policies.act` |

**ACT 权重是 0.3.3 存的，用 0.1.0 加载会报 `No module named 'lerobot.policies'`。**
反过来 pi0.5 那套依赖 0.1.0 的接口。所以两个环境并存，各管各的，别想着合并。

`eval_venv_act` 是由 `eval_venv` 复制后替换 lerobot 得到的——这样仿真侧依赖
（embodichain、robosynchallenge、dexsim 等）不用重装。

## 重建

```bash
WS=$(cd "$(dirname "$0")/.." && pwd)          # 仓库上一级
uv venv --python 3.11 "$WS/venvs/eval_venv_act"
uv pip install --python "$WS/venvs/eval_venv_act/bin/python" \
    "lerobot==0.3.3" "huggingface_hub[cli,hf_xet]" fastapi

# 仿真侧依赖以 editable 方式装入（路径按实际调整）
uv pip install --python "$WS/venvs/eval_venv_act/bin/python" \
    -e "$WS/EmbodiChain" -e "$WS/EmbodiChain/embodichain_tasks" -e "$WS/RoboSynChallenge"
```

系统还需要 `ffmpeg`（LeRobot 的 torchcodec 找不到 libavutil 会直接崩）：
`apt install ffmpeg`。

## 三个坑

### 1. 多卡并行评估不能用 `CUDA_VISIBLE_DEVICES` 指卡

它**只约束 CUDA**；仿真渲染走 Vulkan，仍会枚举全部物理卡并默认选 0 号。两者错位时
会在 `DFGpuSemaphore` 跨 API 导入 semaphore 时直接 abort（实测 8 个分片死 7 个，
只有 shard 0 活着——因为它的 CUDA 卡 0 恰好等于 Vulkan 卡 0）。

正解是用 `eval_policy.py` 的 `--gpu_id N`，它会把 CUDA / JAX / 仿真一并指到同一张物理卡：

```bash
env -u CUDA_VISIBLE_DEVICES "$WS/venvs/eval_venv_act/bin/python" scripts/eval_policy.py \
  --config policy/act/deploy_policy.yml --overrides \
  --task_name <task> --setting random_3p --checkpoint_path <ckpt> --gpu_id <N> ...
```

### 2. `ALL_PROXY` 是 socks 时 httpx 直接报错

`huggingface_hub` 用 httpx，遇到 `socks5h://` 会抛
`ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`。
解法是**只 unset `ALL_PROXY`/`all_proxy`，保留 `HTTPS_PROXY`**——全清掉走直连反而可能超时。

### 3. `--eval_video_log false` 不生效

`--overrides` 的值会走 `eval()`，而 `"false"` 不是合法 Python 字面量（应为 `False`），
解析失败后保留字符串 `"false"`，非空字符串即真值，于是**录像照录**。
要真正关闭得写 `--eval_video_log False`。

## 搬动环境时

venv 里有四类绝对路径会断，按隐蔽程度排序：

1. **editable 安装记录**（`site-packages` 下的 `__editable__*.pth`、`*_finder.py`、
   `direct_url.json`）——最隐蔽，断了报的是下游模块名（如
   `No module named 'embodichain_tasks.tableware'`），看不出是路径问题；
2. **`bin/*` 的 shebang**——直接调 `bin/python` 不受影响，但 `pip` 等控制台脚本会挂；
3. **`bin/python` 指向 uv 托管解释器的软链**——跨机复制时目标机没装对应 CPython 版本就是断链；
4. 自己写的启动脚本里的硬编码路径。

一次性排查：

```bash
for SP in "$WS"/venvs/*/lib/python*/site-packages "$WS"/RoboSynChallenge/policy/*/.venv/lib/python*/site-packages; do
  for f in "$SP"/*.pth; do
    [ -f "$f" ] || continue
    while read -r l; do case "$l" in /*) [ -e "$l" ] || echo "失效: $l  <- $f";; esac; done < "$f"
  done
done
```

**把整个目录搬走再软链回原路径不会断**（绝对路径仍然有效）；改变路径结构才需要大量修补。
搬完必须端到端实跑一次（跑 2–3 集评估），只做 `import` 检查会漏掉运行期才加载的东西。
