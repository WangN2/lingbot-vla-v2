# 多 Agent 协作看板（LingBot-VLA 2.0）

> 固化于 2026-09-26。看板机制自 2026-09-24 起在本机运行，本文将其固化为可复用文档。
> 操作入口：仓库内 skill `/lingbot-board`（`.claude/skills/lingbot-board/SKILL.md`）。

## 1. 背景与目标

本项目的开发由多个 Claude Code agent 会话 + 功能负责人（FO）会话在**同一台服务器、同一个共享 checkout** 上协作。风险：改写冲突、任务重复领取、孤儿工作（开工不登记、写完不提交）、长进程（训练/评测）无人跟踪。

解法：一个**看板文件**作为唯一事实源，配合 git 分支纪律与定时巡检。

- **看板路径**：`/home/likang07/wangbin/lingbot-agent-board.md`（在仓库外，agent 无需写仓库权限即可更新任务状态）
- **适用范围**：仅 lingbot-vla-v2 项目的 agent。其他项目（nio_vln 等）的 agent 不在看板登记，FO 巡检不追踪。

## 2. 看板结构

| 区块 | 内容 | 写入规则 |
|---|---|---|
| 协议 | 所有 agent 必须遵守的 7 条规则 | FO 维护，改协议需在事件日志说明 |
| 状态说明 | 状态码定义 | 同上 |
| 任务表 | id / 标题 / 状态 / 负责人 / 分支提交 / 更新日期 / 备注 | 领取人更新自己的行 |
| 事件日志 | append-only 时间线 | **一行一条，只增不改** |

任务 id 格式 `T-NNN`，由建任务者递增分配。

## 3. 状态机与流转

```
pending ──领取──> in_progress ──提测──> review ──FO通过──> done
                      ^                   │
                      └──── rework <──────┘（FO驳回，按备注返工）
blocked（紧急阻塞，任意状态可切，解除后回原状态）
```

- **pending**：待领取。任何人不可认领非 pending 任务。
- **in_progress**：进行中。领取时"负责人"填会话标识 + 日期。
- **review**：开发完成，附分支名、最新 commit、pytest 结果摘要。**开发者不自行合入**。
- **rework**：FO 审查不通过，备注写明原因。
- **done**：FO 审查通过并已合入 feature/learning。
- **blocked**：紧急阻塞，@FO 升级。

## 4. 分支纪律

1. 任务一律在独立分支开发：`agent/<任务id>-<短名>`（如 `agent/t001-config-drift`）。
2. **禁止直接提交 feature/learning**（FO 合入动作除外）。
3. 共享 checkout：动手前 `git status` 确认干净；发现他人未提交改动勿覆盖，写任务备注并等待。
4. 合入方式：FO 审查通过后合入 feature/learning 并推送（见 §6）。

## 5. FO（功能负责人）职责

- **巡检**：工作时段每 2 小时（9:33–21:33），内容：读看板状态变化 → `git log --all --since="2 hours ago"` 看新提交 → 检查训练/评测进程（`ps aux | grep -E "train_lingbotvla|lingbot_vla_v2_policy|start_robotwin"` + `nvidia-smi`）→ 审查 review 状态任务 → 看板事件日志 append 记录。无变化时飞书可省略，事件日志必记。
- **日报**：每日 10:02，飞书发送项目进度/风险变化/当日计划。
- **审查**：对 review 任务 `git show` 看 diff + 跑全套 pytest（`~/miniconda3/envs/lingbotvla/bin/python -m pytest tests/`），通过才合入。
- **升级**：blocked 超过一个巡检周期、或 in_progress 超一天无提交 → 飞书提醒。
- **孤儿接管**：未登记即开工的工作产物（如工作区出现陌生未跟踪文件），静止超 4.5 小时视为孤儿，FO 按事先宣布的兜底策略接管（走正常分支→审查→合入流程）。接管前先在事件日志公告。
- **长进程跟踪**：训练/评测的磁盘水位、崩溃恢复（enable_resume 机制）由 FO 监控并通报。
- 巡检/日报由 FO 会话的 cron 实现（会话级任务，7 天自动过期，FO 需在过期前续期）。

## 6. git 工作流（本机特有）

远端 `origin` 是 **pull-only 的 ghfast 代理**，推送走专用 SSH 443 通道：

```bash
# 推送（FO 合入后）
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_nio_vln -o IdentitiesOnly=yes' \
  git push ssh://git@ssh.github.com:443/WangN2/lingbot-vla-v2.git feature/learning

# 推送后同步本地 origin 跟踪引用（否则 git status 显示虚假 ahead/behind）
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_nio_vln -o IdentitiesOnly=yes' \
  git fetch ssh://git@ssh.github.com:443/WangN2/lingbot-vla-v2.git \
  feature/learning:refs/remotes/origin/feature/learning
```

开发分支（`agent/*`）为本地分支，审查合入后即删，不推远端。

## 7. 新 agent 入职指令模板

给新开发 agent 会话的开场指令（复制即用）：

> 你是 lingbot-vla-v2 的开发 agent。开始任何工作前，先调用 `/lingbot-board` 技能（或用 Read 工具读看板 `/home/likang07/wangbin/lingbot-agent-board.md`），按看板协议从任务表领取 pending 任务：改 in_progress、建 `agent/<id>-<短名>` 分支开发、自测全绿后改 review 提审，等待 FO 审查合入。禁止直接提交 feature/learning。事件日志只许 append。

## 8. 约定与红线

- **飞书通知**：所有对外消息必须含关键词 `lingbot`；webhook 地址由 FO 会话持有，**不写入仓库**（避免凭证入库）。
- **事件日志 append-only**：历史行禁止改写删除。
- **数据/模型红线**：`lerobot_datasets/`、`RoboTwin/`、`models/`、checkpoint 删除需用户显式授权，FO 未获授权不删任何文件。
- **他人目录**：`peak.li`、`xbliu`、`env_isaac`、nio_vln 等非本项目目录不碰。

## 9. 已验证运行记录

- 2026-09-24 建板；T-006（孤儿接管 benchmarks.md）、T-001（FO 亲自开发 config 漂移防护，48 测试）均走完整流程合入。
- 2026-09-25 训练 20260925_test 全程由巡检跟踪（磁盘 6 次升级预警 → 崩溃恢复 → 30000 步完成）。
- 2026-09-26 协议固化为本文档 + `/lingbot-board` skill。
