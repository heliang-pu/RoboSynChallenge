# 仓库约定：分支、worktree、产物、提交

## 分支分三类，别混

| 分支 | 角色 |
|---|---|
| `main` | **测评分支**，实验分支合流后在这里跑评测；policy 目录最全，`tests/` 与仓库根 `.venv` 只在这条分支 |
| `official/main` | **官方同步分支**，跟踪 `origin/main`（主办方 EDEM-AI），只负责把上游拉进来，不在上面开发 |
| 其余 | 实验分支，各管一摊 |

实验分支：`feat/parallel-eval`（并行评估）、`feat/parallel-collect`（并行采集）、
`sim-recap`（RECAP 价值函数闭环）、`feat/rtc-async-pi05`（实时分块与异步执行）、
`feat/realtime-vla-pi05`（推理加速）、`ppo-post-training`（PPO 后训练）、
`fix/random-spawn-reachability`（收窄够不着的物体生成范围）。

远端：`origin` = 主办方 EDEM-AI（**只读上游**），`mine` = 个人 fork（推这里）。

## worktree

**实验分支各开 worktree，不要在主目录 `git checkout` 它们**（会直接报 already checked out）。
`git worktree list` 是权威清单。要改哪条线就 `cd` 过去。

**各 worktree 有各自的 CLAUDE.md**，按该分支实际有的东西写，不要跨目录照搬。
它们通常没有自己的 `.venv`（软链到 main 的），也没有 `tests/`。

README 里 `<!-- branch-readme:begin/end -->` 之间那段是**每个分支各自维护**的分支说明，
切分支改这块，**不要把它当冲突合掉**——合并时容易把别的分支的块整段带进来，
造成一个 README 里出现两个导航块。

**rsync 过去的仓库副本会带上本机路径的幽灵 worktree 注册**，导致 push 被拒
（`branch is currently checked out`）——先 `git worktree prune`。

## 产物落盘

- 评估产物见 [evaluation.md](evaluation.md)。
- 大产物一律 gitignore：`lerobot_dataset/`、`training_data/`、`eval_result/`、`/report/`、
  `outputs/`、`checkpoints`。**报告类结论要留档就写进 `docs/`**，别指望 `report/` 能被 clone 到。

## 无 lint / format 门禁

`pyproject.toml` 的 `[tool.black]` 是个空节，没有 ruff/pre-commit 配置，`.github/` 只有一个
PR 模板。**别去找「跑一下 lint」的命令**，按周围代码风格写即可。

## 其它

`scripts/eval_policy_parallel.py.fixed` 是未入库的游离副本，真正的入口永远是
`scripts/eval_policy_parallel.py`。
