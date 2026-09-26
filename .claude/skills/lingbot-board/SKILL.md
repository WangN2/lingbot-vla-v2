---
name: lingbot-board
description: lingbot-vla-v2 多 agent 协作看板协议。任何要在 lingbot-vla-v2 仓库开发的 agent 会话，开工前用本技能了解领取任务、分支纪律、自测提审的完整流程；含 FO（功能负责人）巡检/审查/合入职责。
---

# LingBot-VLA 2.0 多 Agent 看板协作

看板文件（唯一事实源）：`/home/likang07/wangbin/lingbot-agent-board.md`
完整协议文档：本仓库 `docs/agent-board.md`

## 你是开发 agent：开工流程

1. **读看板**：用 Read 工具读看板文件。只认任务表里 `pending` 状态的任务。
2. **领取**：把目标任务状态改 `in_progress`，"负责人"填你的会话标识 + 日期。一次只领一个。
3. **建分支**：`git checkout -b agent/<任务id>-<短名>`（如 `agent/t003-fix-init`）。**禁止直接提交 feature/learning**。
4. **共享 checkout 纪律**：动手前 `git status` 确认工作区干净；发现他人未提交的改动不要覆盖，写进任务备注并等待。
5. **开发 + 自测**：`~/miniconda3/envs/lingbotvla/bin/python -m pytest tests/` 全绿才可提测。测试基建在 `tests/`，勿绕过。
6. **提测**：状态改 `review`，"分支/提交"填分支名与最新 commit，备注写 pytest 结果摘要。**不要自行合入**。
7. **等待审查**：FO 每 2 小时巡检并审查（看 diff + 跑全套测试）：通过 → 改 `done` 并合入 feature/learning；不通过 → 改 `rework` 并写原因，按备注返工后重新提 `review`。
8. **沟通**：澄清问题写在任务备注列并 @FO；紧急阻塞把状态改 `blocked`。
9. **长任务登记**：启动训练/评测等长进程时，在任务备注写明输出目录、命令与预计时长——FO 巡检要跟踪。

## 状态机

`pending`（待领取）→ `in_progress`（进行中）→ `review`（待审）→ `done`（已合入）｜ `rework`（返工）｜ `blocked`（阻塞）

## 硬性纪律

- 事件日志（看板末尾）**只许 append**，一行一条，禁止改写或删除历史行。
- 任务表除自己领取/提测的字段外，不要动他人任务的行。
- 飞书 webhook 地址由 FO 会话持有，**不写入仓库**。
