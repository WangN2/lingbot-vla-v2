# 具身智能训练数据集与评测基准调研（操作类 / 移动类 / 世界模型 / 人形全身）

> 调研时间：2026-09。覆盖范围：机械臂操作 VLA、机器人/视频世界模型、移动导航（VLN/ObjectNav/四足与人形移动）、人形全身与移动操作、家庭服务机器人。
> 调研方式：5 个方向并行检索，arXiv 编号经论文页逐条核验；无法二次核实的数字标 ⚠️ 待核实（汇总见 [§7](#7-待核实清单)）。
> 校验：2026-09-24 对正文 30+ 个 arXiv 编号经 arxiv.org 直接抽查（含全部 2026 年新编号），**全部属实**；RoboTwin 2.0 的 50 任务/731 物体/147 类、AgiBot World 的 100 万+轨迹/217 任务、HM3D 1000 场景等承重数字亦经摘要页确认。
> 素材来源：`tmp/bench_survey/01~05`（gitignored 原始调研稿），本文档为唯一对外口径。
> 用途：为 LingBot-VLA 2.0 后续训练数据扩充与评测基准接入提供选型依据（适配建议见 [§6](#6-对-lingbot-vla-20-的适配建议)）。

---

## 1. 领域地图（总览）

| 方向 | 代表工作 | 训练数据主原料 | 评测主基准 | 数据量级（2026 现状） |
|---|---|---|---|---|
| 操作类 VLA | RT-1/2、π0 系、OpenVLA、RDT-1B、GR00T、CogACT | OXE/DROID 等真机轨迹 + 互联网/人类视频 + 仿真 | LIBERO、CALVIN、SimplerEnv、RoboCasa、RoboTwin、真机 | 真机 100 万轨迹级 / 1 万小时级（π 系私有） |
| 世界模型（机器人向） | Cosmos、DreamGen、V-JEPA 2、iVideoGPT、1X WM | 大规模无动作视频预训练 + 小量 action-conditioned 机器人数据 | FVD/LPIPS、Physics-IQ、DreamGen Bench、1X Challenge | 互联网视频 2000 万小时级；机器人数据 60h~5700h 后训练 |
| 移动/导航 | NaVid 系、NaVILA、ViNT/NoMaD、VLFM | 仿真指令集（R2R/RxR）+ 扫描场景（HM3D/MP3D）+ 真机/人类视频轨迹 | R2R-CE、RxR、HM3D ObjectNav、GOAT-Bench、VLN-CE-Isaac/HumanoidVLN | 百万样本级（Uni-NaVid 3.6M）、830k VLN 数据（InternData-N1） |
| 人形全身/移动操作 | GR00T N1~N2、Helix、GO-1、RDT、π0.5 | 人形真机采集场（AgiBot）+ 合成管线（DreamGen/DexMimicGen）+ 动捕 | RoboCasa、RoboTwin、DexMimicGen、BEHAVIOR-1K、真机套件 | 真机人形 100 万轨迹（AgiBot β）+ 仿真 78 万条（DexMG） |

**一句话现状**：操作类与导航类各自形成了「互联网视频 action-free 预训练 → 小量带动作数据后训练 → 仿真/真机评测」的收敛配方；数据格式收敛到 LeRobot（社区/国内系）与 RLDS（谷歌系）双标准；评测上 LIBERO 已近饱和，SimplerEnv/RoboTwin/RoboCasa 承担主要横评，世界模型评测（Physics-IQ 等）刚刚起步。

---

## 2. 训练数据集汇总

### 2.1 真机操作数据（臂式/双臂）

| 数据集 | 规模 | 本体/内容 | 格式 | License | 获取 | 主要使用方 |
|---|---|---|---|---|---|---|
| Open X-Embodiment (OXE) | 100 万+ 轨迹，21 家机构，60+ 子数据集，527 技能/160,266 任务 | 22 种本体（Kuka、WidowX、Franka、Jaco、UR5 等） | RLDS（TFDS `openx_embodiment`） | 各子集不一 | x-embodiment.github.io、HF | RT-X、Octo、OpenVLA、CogACT、RDT（46 子集）、π0、GR00T N1（选 7 子集）、HPT |
| — Fractal（RT-1 私有） | 130k episodes、700+ 任务、17 个月 | 13 台 Kuka 移动操作臂（厨房） | RLDS | 未公开 | 不公开 | RT-1/RT-2、Q-Transformer |
| — BridgeData V2 [2308.12952] | 60,096 轨迹、24 环境、10.7 万语言指令 | WidowX-250 6-DOF | RLDS | CC 系 ⚠️ | 随 OXE | Octo、OpenVLA、SimplerEnv |
| DROID [2403.12945] | 76k episode / 350h+、564 场景、84 任务、12 栋建筑、13 个月分布式 | Franka Panda（12 台）+ Franka 手爪，RGB+深度+力矩，Meta Quest 遥操 | RLDS（社区有 LeRobot 转换版，另有 ~300 episode mini 版） | CC 系 ⚠️ | droid-dataset.github.io | HPT、GR00T N1（140k）、V-JEPA 2-AC（62h 无标签版）、DreamGen（428h） |
| RoboMIND [2412.13877] | 107k 轨迹、479 任务、96 物体类、5k 失败示范 + Isaac Sim 数字孪生 | 4 本体：Franka Panda、UR5e、**AgileX 双臂**（Cobot Magic）、灵巧手人形 | HDF5 | gated ⚠️ | HF `x-humanoid-robomind/RoboMIND` | BAAI 系、国内 VLA 训练 |
| RoboMIND 2.0 [2512.24653] | 310k 双臂轨迹、739 任务、6 本体、12k 触觉 episode、20k 移动操作轨迹 | 同上扩展 | 同上 | 发布中 ⚠️ | 同上 | MIND-2 分层系统 |
| ALOHA / Mobile ALOHA | ALOHA demo 数百条；Mobile ALOHA 866 条（每任务 50） | ALOHA 双臂 / +Tracer 底盘 | HDF5（HF 有 LeRobot 转换版） | 开源 | 项目站、HF | ACT、Diffusion Policy、π 系、co-training 社区 |
| LeRobot 社区数据 | SmolVLA 收 481 数据集 / 22.9k episodes / 10.6M 帧（总量持续增长） | SO-100/101、Koch 等低成本臂 | **LeRobot 原生**（HF Hub 可流式） | 各异 | HF Hub（lerobot 生态） | SmolVLA、社区基准 |
| π 系自有 fleet 数据 | 10,000+ 小时、7 构型、68 任务；π0.5 加 ~400h 家庭数据 | ALOHA/双臂 UR5e/Franka/Mobile ALOHA 等 | 私有 | 不公开 | openpi 只开源权重 | π0/π0.5/π*0.6 |

### 2.2 真机人形 / 全身数据

| 数据集 | 规模 | 本体/采集方式 | 格式 | License | 获取 | 主要使用方 |
|---|---|---|---|---|---|---|
| AgiBot World α/β [2503.06669] | 1,001,552 轨迹 / 2,976.4h / 217 任务 / 87 技能 / 106 场景（α 公开约 14% ≈ 14 万条） | 100 台 AgiBot G1 同构（双 7-DoF 臂+升降腰+底盘+6-DoF 灵巧手/视触觉）；VR 头显+全身动捕双系统，人工在环质检 | HDF5/LeRobot 双格式，9 路相机 | **CC BY-NC-SA 4.0（禁商用）** | HF `agibot-world/AgiBotWorld-Alpha/Beta`（gated） | GR00T N1（α）、N1.5（β）、GO-1（β）、RDT |
| AgiBot World 2026 | G2 平台真机、商用/家庭场景、三层标注（子任务段/2D bbox/技能步骤）、1K–10K episodes 起持续更新 | 同上；自由采集模式 | **LeRobot v2.1**（Parquet+MP4） | CC BY-NC-SA 4.0 | HF + GenieSim 数字孪生 | 2026 社区采用中 |
| AgiBot World Colosseo（评测集） | 100 长程任务、500+ 评测 episode | 同上 | — | — | 官网/GitHub OpenDriveLab/AgiBot-World | CVPR 2025 workshop 挑战赛 |
| RoboMIND（人形子集） | 见 2.1 | 灵巧手人形 + AgileX 双臂 | HDF5 | gated | HF | 同上 |
| AMASS | 40h / 11k+ 动捕序列（光学动捕聚合） | 人类动捕（需 retarget 到人形） | 原始 | 学术许可 | amass.is.tue.mpg.de | HumanPlus、OmniH2O、HOVER、ExBody/2 |
| OmniH2O-6 | 6 个全身任务（RGBD+运动/关节目标 30Hz） | Unitree H1，VR 遥操 | 原始 | 开源 | 项目仓库 | OmniH2O 系 |
| HumanPlus 管线 | 真机 shadowing 遥操数据（每任务 ≤40 demo） | H1 基座 33-DoF + 灵巧手 | 原始 | 开源 | 项目仓库 | HumanPlus |
| TWIST2 数据 [2511.02832] | 全身数据（100 demo / 15min 采集速率） | PICO4U VR 免动捕全身遥操 | 开源 | 开源 | twist-data.github.io | TWIST2 分层策略 |
| 1X 世界模型挑战数据 | 100h+ EVE 人形视频（含 tokens 与 raw actions） | 1X EVE | 视频+动作 | 挑战赛条款 | HF `1x-technologies/worldmodel` | 1X WM 社区/挑战赛 |
| Figure Helix 私有数据 | ~500h 高质量真机遥操 | Figure 02 全上身，多操作员多机 | 私有 | 不公开 | — | Helix |
| GR00T 数据栈 | 真机遥操 88h（VIVE+Xsens）；DreamGen 神经轨迹 827h（10× 放大）；N1.7 加 2 万小时 EgoScale 人类视频 | Fourier GR-1 等 | LeRobot v2 变体 + modality.json | **Apache 2.0（含 demo 数据）** | GitHub Isaac-GR00T + HF | GR00T N1/N1.5/N1.7 |

### 2.3 移动/导航数据

| 数据集 | 规模 | 内容/来源 | 格式 | License | 获取 | 主要使用方 |
|---|---|---|---|---|---|---|
| R2R [1711.07280] | 7,189 轨迹 / 21,567 指令 | MP3D 61 场景离散导航图，真人标注（VLN 指令） | 官方格式 | 学术 | 官网 | NaVid、Uni-NaVid、NavGPT 系 |
| RxR [2010.07954] | 16,522 路径 / 126k 指令（英/印地/泰卢固） | MP3D，含稠密时空标注；RxR-Habitat 连续版（Google） | 官方格式 | 学术 | 官网 | NaVILA、Uni-NaVid |
| REVERIE [1904.10151] | 21,702 指令 / 4,140 目标物体（489 类） | 90 MP3D 建筑，导航+远程指认 | 官方格式 | 学术 | 官网 | REVERIE 系 |
| VLN-CE（R2R_VLNCE_v1-3）[2004.02857] | R2R/RxR 连续动作空间版 | Habitat，MP3D 90 场景 | Habitat episodes | 学术 | Google Drive | NaVid/NaVILA/Uni-NaVid/Open-Nav |
| EnvDrop / ScaleVLN [2307.15644] | speaker 合成增强指令；ScaleVLN 大规模扩充 R2R | HM3D/Gibson 新场景 | 同上 | 学术 | 项目仓库 | NaVILA、NavGPT-2 |
| NaVILA-Dataset | 2K YouTube 游览视频 → 20K 轨迹（MASt3R 位姿→动作，VLM 生成指令）+ r2r/rxr/envdrop/scanqa/human | 人类视频转导航数据 | HF 数据集 | 开源 | HF `a8cheng/NaVILA-Dataset` | NaVILA |
| GNM/ViNT/NoMaD 真机轨迹集 | 60–160h 真实导航轨迹（11 源、8 平台：SCAND/SACSoN/TartanDrive/NeBula/RECON/BDD 等） | 室内/校园/越野/人行道 | 各源格式 | 各异 | GitHub robodhruv/drive-any-robot | GNM/ViNT/NoMaD |
| HM3D / HM3DSem [2109.08238] | 1,000 个真实住宅扫描 | 室内重建（ObjectNav 底座） | Habitat 格式 | 学术（需 Matterport 授权） | 官网申请 | VLN-CE、GOAT-Bench、Uni-NaVid、VLFM |
| InternData-N1 | 3k+ 场景、830k VLN 数据 | 多具身多场景 | InternNav 工具箱 | 学术 | 上海AI Lab | 2025 年最大开源导航数据 |
| MP3D / Matterport3D | 90 栋建筑 PBR 扫描 | 室内 RGB-D 全景 | Matterport 格式 | 学术 | 官网申请 | R2R/RxR/REVERIE 底层 |
| RoboTHOR / ProcTHOR | 89 公寓（60/15 train/val）；ProcTHOR 程序化 10k 房屋 | AI2-THOR 交互场景 | AI2-THOR | MIT 系 | ai2thor.com | ObjectNav 系、ESC |

### 2.4 仿真/合成数据管线（生成型）

| 数据集/管线 | 规模 | 平台/生成方式 | 获取 | 主要使用方 |
|---|---|---|---|---|
| RoboTwin 2.0 数据管线 [2506.18088] | 50 任务自动生成大规模 demo（合成数据+10 条真机 demo 比纯 10-demo 相对 +367%） | ManiSkill3 数字孪生 + 域随机化 + MLLM 自动写专家代码 | 开源（HF 双臂） | RoboTwin 2.0、双臂 VLA（含本项目） |
| RoboCasa 数据 [2406.02523] | 100 任务（25 原子+75 LLM 复合）、100k 轨迹 | robosuite/MuJoCo + MimicGen 扩增，Franka+Omron | robocasa.ai | RoboCasa、GR00T N1 系、DreamGen |
| DexMimicGen | 78 万条 / 6500h（**11 小时机时生成**） | robosuite + 灵巧手，少量源 demo 自动变换 | NVIDIA 开源 | GR00T N1、灵巧手系 |
| MimicGen | 5 万级生成轨迹（源 demo 数百条） | robosuite/MuJoCo | NVIDIA 开源 | MimicGen、RoboCasa |
| DreamGen 神经轨迹 [2505.12705] | RoboCasa 240k 合成轨迹、GR-1 22 个新行为（88h 真机→827h） | 视频世界模型生成 + LAPA/IDM 伪动作标注 | 代码/bench 开源 | GR00T 系 |
| LIBERO 训练数据 | 4 套件各 10 任务 × 50 demo | robosuite/PyBullet，Franka | libero 站 | LIBERO、OpenVLA 微调、WorldVLA |
| CALVIN 训练集 [2112.03227] | 24h 遥操 play、22,966 轨迹、34 任务（语言稀疏标注） | PyBullet，Franka | calvinrobot.github.io | GR-1、LCB、OpenHelix |
| GenieSim | AgiBot 数字孪生（与真机 1:1） | 自家仿真 | 随 AgiBot World 2026 | AgiBot 生态 |

### 2.5 互联网/人类视频语料（VLM 与世界模型预训练）

| 语料 | 规模 | 用途 | License | 获取 |
|---|---|---|---|---|
| Ego4D | 3,670h、9 国 74 场景、170 万句叙述 | 第一人称视频预训练（GR-1 3500h、GR00T 2145h、NaVILA 游览视频源、UniSim 3.5M clips） | 学术 | ego4d-data.org |
| Epic-Kitchens-100 | 100h、45 厨房、~9 万细粒度片段 | 第一人称动作理解（UniSim、V-JEPA 2） | CC 系 | epic-kitchens.github.io |
| Something-Something V2 | 220,847 视频、174 动作类 | 时序物理/人手操作（iVideoGPT、VPP、DreamGen） | 研究许可 | 官网 |
| YouTube-Temporal-1B / Kinetics / HowTo100M | V-JEPA 2：1.4M h 检索净化为 115M 场景；HowTo100M 1.36 亿 clip（已收缩） | 大规模 action-free 视频预训练 | 各异 | 各官网 |
| GR-2 网络视频 | 3,800 万片段 / 50B+ token | 视频预训练→操作微调 | 私有切片 | ByteDance |
| EgoScale（N1.7） | 2 万小时人类视频 | 人/机器人共享相对 EEF 空间直接预训练 | NVIDIA 内部 | — |

### 2.6 世界模型训练数据（专用配方）

| 模型/管线 | 训练数据 | 说明 |
|---|---|---|
| Cosmos-Predict 1/2.5 | 20M h 网络视频 → 清洗为 100M~200M clips；后训练：1X EVE 200h（action-conditioned）、RDS 20k h 驾驶 | 「海量无动作预训练 + 小量动作后训练」样板 |
| V-JEPA 2 | >1M h 互联网视频（action-free）+ **仅 62h 无标签 DROID**（-AC 后训练） | 62h 动作数据即零样本 Franka 操作 |
| DreamGen | LAPA 混合 438.1M 帧 / 5,721.3h：AgiBot-Alpha 1979h、Ego4D 2145h、DROID 428h、RoboCasa 268h、RT-1 338h、Language Table 196h、Bridge-v2 111h、GR-1 88h、DexMG 62h、SSv2 106h | WM 生成合成数据的训练配方 |
| iVideoGPT | OXE 35 子集 + SSv2 共 1,417,954 轨迹（action-free 预训练） | 机器人域视频预训练 |
| Vidar | AgiBot-World + RoboMIND + RDT 共 746,533 episodes（三相机双臂统一观测） | 视频扩散+逆动力学 |
| UniSim | RT-1 70k + Bridge 2k + Language Table 600k + 机器人杂项 133k + Ego4D 3.5M + SSv2 160k + EPIC 25k + HM3D/R2R + LAION | 交互式真世界模拟器 |
| GAIA-1/2（自驾） | 4,700h 伦敦真车 / 25M 条 2s 序列 | 私有（WayveScenes101 公开） |

---

## 3. 评测基准汇总

### 3.1 操作类仿真基准

| 基准 | 任务规模 | 平台 | 指标 | 代表水平 | 获取 | 谁在用 |
|---|---|---|---|---|---|---|
| **LIBERO** [2306.03310] | 130 任务 4 套件（Spatial/Object/Goal/Long 各 10+LIBERO-100） | robosuite（MuJoCo），Franka 单臂 | 成功率 | OpenVLA 76.5% → OFT **97.1%**（近饱和） | libero 站；微调用 `physical-intelligence/libero` | OpenVLA 系、π 复现、SmolVLA、GR00T N1.7 |
| **SimplerEnv** [2405.05941] | Google Robot（Fractal）4 任务 + Bridge/WidowX 变体；Visual Matching 协议 | SAPIEN/ManiSkill | 成功率 | CogACT WidowX 61.3% vs OpenVLA 39.3%；RT-1/Octo 横评 | github.com/simpler-env/SimplerEnv | Octo、OpenVLA、CogACT、RoboVLMs、GR00T N1.7 |
| **CALVIN** [2112.03227] | 34 任务、5 步长程链、ABCD→D 泛化 split | PyBullet，Franka | 平均连续成功指令数（Avg.Len 0~5） | GR-1 94.9%（1% 语言）；VPP 4.33；OpenHelix 双系统 SOTA | calvinrobot.github.io | GR 系、LCB、VPP、OpenHelix |
| **RoboCasa** [2406.02523] | 100 厨房任务（GR00T 用 24） | robosuite | 成功率 | GR00T N1 30demo 17.4% → N1.5 47.5% | robocasa.ai | GR00T 系、Diffusion Policy |
| **RoboTwin 1.0/2.0** [2504.13059]/[2506.18088] | 2.0：50 双臂任务、5 本体、731 物体/147 类、5 轴域随机化、clean/randomized 双模式 | ManiSkill3/SAPIEN | 成功率（分模式） | 本项目 LingBot-VLA 2.0：clean 93.52% / randomized 92.80% | 项目仓库（HF） | 双臂 VLA 事实标准（ICML 2026） |
| **ManiSkill3** | 数十任务族（GPU 并行加速） | SAPIEN | 成功率/SPL | — | maniskill.ai | ManiSkill 挑战赛社区 |
| **RLBench** | 100 任务 | CoppeliaSim | 成功率 | — | 项目仓库 | PerAct、RVT、RoboDreamer |
| **MetaWorld** | 50 任务（MT-50） | MuJoCo，Sawyer | 成功率 | — | 项目仓库 | 多任务 RL、SmolVLA、HPT |
| **BEHAVIOR-1K / HAB** [2403.09227] | 1000 日常活动、50 场景、9000+ 物体 | OmniGibson | 任务完成度/BPS | 真机基线仍低（OVMM ~20%） | 官网 | Stanford 系、M3 |
| **DexMimicGen** | 灵巧手任务族 | robosuite | 成功率 | GR00T N1 基准之一 | NVIDIA 开源 | GR00T 系、灵巧手系 |

### 3.2 导航类基准

| 基准 | 任务规模 | 平台 | 指标 | 代表水平 | 获取 |
|---|---|---|---|---|---|
| **R2R-CE（VLN-CE）val-unseen** | 1,839 eps / 11 场景 | Habitat 连续控制 | SR/SPL/OSR/NE/TL | 监督 SOTA ≈70–75；视频 VLA：NaVid 41.9 → Uni-NaVid 51.8 → NaVILA 49.7 | VLN-CE 官方 |
| **RxR(-CE)** | 16,522 路径（3 语言） | MP3D/Habitat | SR/nDTW | NaVILA 零样本 SR 23–54 | 官网 |
| **HM3D ObjectNav** | val 2,000 eps / 20 场景 / 6 类 | Habitat | SR/SPL（1m 内停） | Uni-NaVid SR 73.7 / SPL 37.1（纯 RGB 超 PIRLNav-RL 70.4） | Habitat 挑战赛 |
| **GOAT-Bench** [2404.06609] | 181 HM3DSem 场景、312 类、多模态目标（类别/语言/图像）、500 步预算 | Habitat | SR/进度 | — | 项目仓库 |
| **RoboTHOR** | 89 公寓、12 类小物体、val 1,800 eps | AI2-THOR | SR/SPL | — | ai2thor.com |
| **VLN-CE-Isaac / NaVILA-Bench** | 1,077 轨迹 | Isaac Sim（带物理四足/人形） | SR/SPL | Habitat 运动学步进不真实 → 物理仿真 | NaVILA 项目 |
| **HumanoidVLN** [2608.12860] | 87 场景、933 eps、4 具身（G1/H1 等） | Isaac Sim + 3DGS Real2Sim | SR/SPL/nDTW + **Fall Rate** | H1 上 VLN 模型摔倒率 64–71%；最优 JanusVLN SR 43.55 | 论文 |
| **EVT-Bench** | 25,986 eps / 804 场景 / 100 虚拟人 | 仿真 | 视觉跟踪指标 | TrackVLA 自建 | 论文 |
| **HM3D-OVON**（开放词汇） | — | Habitat | SR/SPL | Uni-NaVid 零样本 ≈40 | Habitat 挑战赛 |

### 3.3 真机评测

| 套件 | 任务 | 平台/本体 | 指标 | 谁在用 |
|---|---|---|---|---|
| ALOHA / Mobile ALOHA 真机 | 6+ 精细双臂任务；Mobile 系烹饪/家务 | 双臂真机 | 成功率 | ACT、DP、π0、OFT |
| OpenVLA Bridge 真机 | 8 个 WidowX 桥任务 | WidowX | 成功率 | OpenVLA、Octo 系 |
| π 系真机协议 | 折衣/收桌/组盒/倒咖啡；π*0.6 咖啡 **连续 13h** | ARX/Franka/Koch 多本体 | 成功率/时长/鲁棒性 | π 系 |
| RT-1/2 真机 | RT-1 700+ 任务 | Google 厨房机队 | 成功率 | RT 系 |
| Helix 真机 | 数千未见家庭物品零样本抓放 | Figure 02 | 零样本成功率 | Figure |
| π0.5 家庭评测 | 全新家庭厨房/卧室清理（长程） | 移动双臂 | 端到端完成率 | PI |
| Go2 四足真机 | NaVid 8 环境 SR 40–92%；LOVON 开放词汇目标 | Unitree Go2/B2/H1-2 | SR | 北大系/USTC 系 |
| HumanPlus 6 任务 | 穿鞋站立/仓储卸货/叠衣/打字等 | 自研 33-DoF 人形 | 成功率 60–100% | Stanford |

### 3.4 世界模型/视频评测

| 基准/指标 | 内容 | 指标 | 谁在用 |
|---|---|---|---|
| FVD / FID / IS / CLIP-S | 视频分布距离与质量（通用协议） | 分布距离 | UniSim、Genie、Cosmos、WorldVLA |
| PSNR / SSIM / LPIPS | 帧级保真（1X Challenge 取第 77 帧预测 vs GT） | 重建误差 | 1X Challenge、iVideoGPT |
| **Physics-IQ** [2501.09038]（+Verified [2606.18943]） | 真实物理实验视频 vs 生成视频、5 领域+反事实 | 物理 correctness/preference | DeepMind、Cosmos 3 |
| **DreamGen Bench** | instruction-following + physics-alignment（VLM-judge + 人评） | IF/PA，与下游 policy 成功率正相关 | DreamGen |
| world-model-as-simulator | 在 WM 内对 policy 排名/评测 | 1X Evaluation 轨 | 1X、V-JEPA 2（零样本 goal planning） |
| RoboDreamer 协议 | 机器人域视频预测 + RLBench 74 任务规划成功率 | FVD + 规划成功率 | RoboDreamer |

---

## 4. 论文 → 数据集映射（按方向精简表）

### 4.1 操作类 VLA

| 模型 | 机构/年份 | 训练数据 | 评测基准与成绩 |
|---|---|---|---|
| RT-1 [2212.06817] | Google 2022 | Fractal 130k eps（私有）+ 仿真 | 真机 700+ 任务 97% |
| RT-2 [2307.15818] | DeepMind 2023 | VLM 互联网图文 + RT-1 数据 | 真机涌现语义推理 |
| RT-X/OXE [2310.08864] | 21 机构 2023 | OXE 100 万轨迹 22 本体 | RT-1-X +50%、RT-2-X 涌现 2× |
| Octo [2405.12213] | Berkeley 等 2024 | OXE 子集 800k 轨迹 | 9 真机平台零样本/微调 |
| π0 [2410.24164] | PI 2024 | 自采 10,000h + OXE | 真机叠衣/收桌/装箱 |
| π0.5 [2504.16054] | PI 2025 | +400h 家庭数据 + web | 全新家庭长程清理 |
| π*0.6 [2511.14759] | PI 2025 | +自主数据 + RECAP RL | 咖啡 13h 连续 |
| OpenVLA [2406.09246] | Stanford 等 2024 | OXE 970k 轨迹 | Bridge 真机 +16.5% vs RT-2-X；LIBERO 76.5% |
| OpenVLA-OFT [2502.19645] | UCLA 2025 | OpenVLA 微调配方 | LIBERO **97.1%**、吞吐 26× |
| SmolVLA [2506.01844] | HF 2025 | 481 LeRobot 社区数据集 22.9k eps | 媲美 10× 大模型 |
| RDT-1B [2410.07864] | 清华/AgiBot 2024 | 46 数据集 100 万轨迹 21TB + 自采 6k eps | 真机双臂超 ACT/OpenVLA；1–5 demo 学新技能 |
| CogACT [2411.19650] | BAAI/清华 2024 | OXE（Octo 同款子集） | SimplerEnv 仿真 +35%/真机 +55% vs OpenVLA |
| GR00T N1 [2503.14734] | NVIDIA 2025 | 见 §2.2 数据栈（8,376h 混合金字塔） | RoboCasa 17.4%、DexMG、GR-1 真机 |
| GR00T N1.5 | NVIDIA 2025 | +AgiBot-β + FLARE | RoboCasa 47.5%；真机 83.0% |
| GR00T N1.7 | NVIDIA 2026 | +EgoScale 2 万小时人类视频（共享 EEF 空间） | RoboCasa/SimplerEnv/LIBERO/DROID/真机 G1 |
| GR-1/GR-2 | ByteDance 2023/24 | Ego4D 3500h / 网络视频 3800 万片段 + 小量真机 | CALVIN 94.9%；100+ 任务 97.7% |
| HPT [2409.20537] | MIT/清华 2024 | 52 异构数据集 | 仿真+真机 +20% |
| UniAct [2501.10105] | 北大 2025 | 统一动作空间多本体 | 0.5B 超 14× 大模型 |
| DP3/iDP3 | 上交大等 2024 | 10–40 demo/任务；iDP3 上身遥操 | 仿真 85%；iDP3 25-DoF 人形 |

### 4.2 世界模型

| 模型 | 机构/年份 | 训练数据 | 评测方式 |
|---|---|---|---|
| Genie/2/3 | DeepMind 2024/25 | 200k h 游戏视频→6.8M clips（无动作，LAM） | FVD+定性+实时交互 |
| Cosmos-Predict 1/2.5 | NVIDIA 2025 | 100M~200M clips + 1X EVE 200h 等 | FVD/3D 一致性/物理对齐 |
| Cosmos-Transfer 1/2.5 | NVIDIA 2025 | Cosmos + 空间条件（AgiBot 视频等） | Sim2Real 数据富化、GB200 实时 |
| Cosmos-Reason 1 | NVIDIA 2025 | HoloAssist 166h + 私有 70h | 物理 QA |
| V-JEPA 2 [2506.09985] | Meta 2025 | 1M h 视频 + 62h DROID | SSv2/EK100 + Franka 零样本 |
| DreamGen [2505.12705] | NVIDIA 2025 | 5,721h LAPA 混合 | DreamGen Bench + 下游成功率 |
| iVideoGPT [2405.15223] | 清华 2024 | OXE 35 子集 + SSv2 1.42M 轨迹 | BAIR/RoboNet FVD、MBRL |
| VPP [2412.14803] | 上海AI Lab 2024 | OXE 179k 轨迹 + SSv2 | CALVIN +18.6%、真机 +31.6% |
| WorldVLA [2506.21539] | 阿里 2025 | LIBERO-90 | LIBERO 成功率 +4% 同时 FVD −10% |
| Vidar [2507.12898] | 清华 2025 | AgiBot+RoboMIND+RDT 746k eps | 跨任务/背景/相机泛化 |
| UniSim [2310.06114] | Google/UCB 2023 | 机器人+人类视频+图文混合 | FVD/FID + 模拟器内训 VLA |
| 1X WM / Challenge [2510.07092] | 1X 2024/25 | EVE 数千小时（challenge 100h 开放） | PSNR@77/SSIM/LPIPS/FID + policy 排名 |

### 4.3 移动/导航

| 模型 | 机构/年份 | 训练数据 | 评测基准与成绩 |
|---|---|---|---|
| NaVid [2402.15852] | 北大+Galbot 2024 | R2R-CE 510k 样本（纯 RGB） | R2R-CE SR 41.9；Go2 真机 8 环境 |
| Uni-NaVid [2412.06224] | 北大+Galbot 2024 | 3.6M 样本（R2R/RxR/HM3D/EQA/跟人） | R2R-CE 51.8；HM3D ObjectNav 73.7 |
| NaVILA [2412.04453] | UCSD+NVIDIA 2024 | VLN-CE + 20k YouTube 轨迹 | R2R-CE 49.7；Go2/H1/Booster 真机 |
| TrackVLA [2505.23189] | 北大 2025 | 855K×2 跟踪/识别样本 | EVT-Bench；Go2 真机 |
| ViNT [2306.14846] / NoMaD | Berkeley 2023 | 150h+/100h 真机轨迹 11 源 | LoCoBot/Go1/Jackal 真机 |
| GNM [2210.03370] | Berkeley 2022 | 60h/6 平台/8 数据集 | 多平台零样本迁移 |
| VLFM [2312.03275] | GaTech+Meta 2023 | 免训练（RGB-D） | HM3D SR 52.5（近监督） |
| NavGPT-2 [2407.12366] | Adelaide 等 2024 | 10k GPT-4V 标注 + R2R | R2R SR 67.5 |
| Mobility VLA [2407.07775] | Waymo+GDM 2024 | 1 条巡游视频 + Gemini | 办公室 SR 80–85% |

### 4.4 人形全身与移动操作

| 模型 | 机构/年份 | 训练数据 | 评测基准与成绩 |
|---|---|---|---|
| GR00T N2 | NVIDIA 2025（闭源） | UGRPO 仿真 RL（Isaac Lab） | 1X NEO 商用部署 |
| Figure Helix | Figure 2025（闭源） | ~500h 真机遥操 | 零样本家庭抓放、双机协作 |
| OpenHelix [2505.03912] | 西湖大学 2025 | CALVIN | CALVIN ABC-D 双系统 SOTA |
| GO-1 [2503.06669] | 智元 2025 | AgiBot β 100 万 + 网络视频（latent action） | 复杂灵巧任务 >60%（+32% vs RDT） |
| HumanPlus [2406.10454] | Stanford 2024 | AMASS + shadowing 遥操 | 6 真机任务 60–100% |
| OmniH2O [2406.08858] | CMU 等 2024 | AMASS retarget + VR 遥操 | 全身运动+操作 |
| HOVER [2410.21229] | NVIDIA/CMU 2024 | 多教师蒸馏（AMASS 系） | 多模式全身控制 |
| ExBody/2 | UCSD 2024 | 人类动捕+仿真 | G1 上超 4 基线 |
| iDP3 [2410.10803] | 上交大等 2024 | AVP 上肢遥操自采 | 单场景训练跨场景泛化 |
| Mobile ALOHA [2401.02117] | Stanford 2024 | 50 demo/任务 + 静态 ALOHA co-training | 真机烹饪/家务；co-training +90% |
| TidyBot/++ [2305.05658]/[2412.10447] | Princeton 等 2023/24 | 手机 App 遥操 + LLM 个性化 | 真机家庭整理 |
| HomeRobot [2306.11565] | AI2/Meta 2023 | OVMM 基准 | 真机基线 20% |

---

## 5. 2025–2026 趋势要点（合并去重）

**数据侧**
1. **真机数据工厂化**：AgiBot World 100 台同构机同场采集达 100 万轨迹/2976h；DROID 12 址分布式 76k eps；human-in-the-loop 质检成为标配。头部壁垒 = 采集场产能（π 系 1 万小时私有仍是最大护城河）。
2. **合成数据引擎大爆发**：DreamGen 视频生成 88h→827h（10×）；DexMimicGen 11 机时生成 78 万条；RoboTwin 2.0 域随机化+MLLM 写专家代码；「世界模型造数据」从视觉增广升级为物理可行演示生成（2026 RoboDream）。
3. **「大规模 action-free 视频预训练 + 小量动作后训练」配方定型**：V-JEPA 2（1M h + 62h DROID）、Cosmos（20M h + 200h 1X）均验证；Genie latent action 思想被 LAPA/CLAP 发展为动作自监督。
4. **人类视频预训练常态化**：GR-1（Ego4D）→ GR-2（3800 万片段）→ GR00T N1.7（EgoScale 2 万小时，人/机共享相对 EEF 空间）→ π0.5（web 数据）。
5. **动作空间统一成为跨本体关键**：RDT 128 维物理量纲 / GR00T N1.7 相对 EEF / GO-1 latent action（VQ-VAE）。
6. **数据来源从演示转向经验**：π*0.6 RECAP 自主执行+纠错 RL；2026 Fleet-Scale RL；GR00T N2 仿真 RL（UGRPO）直接商用部署。
7. **导航数据人类视频化**：NaVILA YouTube 2K→20K 轨迹（MASt3R 位姿+VLM 指令）；Ego4D 支撑表征预训练。
8. **格式收敛 LeRobot**：AgiBot 2026（v2.1）、GR00T（v2+modality.json）、SmolVLA 社区 481 数据集；RLDS 仍是谷歌系标准；HF 为分发中心。

**评测侧**
9. **操作基准分层**：LIBERO 近饱和（OFT 97.1%）→ SimplerEnv（真对齐横评）、RoboCasa/RoboTwin（双臂/家庭/随机化）承担主横评；新增物理理解（RoboTwin-Phys）、量化（VLAQuantBench）等维度。
10. **导航评测物理化**：Habitat 运动学步进不真实 → VLN-CE-Isaac/HumanoidVLN 引入摔倒率/碰撞率（H1 摔倒率 64–71%），人形导航显著难于四足。
11. **真机评测成标配且走向长程/鲁棒指标**：Go2/H1/Booster/Spot 直接部署；π*0.6 以「连续 13 小时」计；π0.5 全新家庭端到端。
12. **世界模型评测从 FVD 转向物理一致性与闭环**：Physics-IQ(+Verified)、DreamGen Bench（与下游成功率正相关）、1X Evaluation 轨（WM 内 policy 排名）。
13. **评测滞后于数据**：家庭长程基准（BEHAVIOR-1K/OVMM）真机基线仍 ~20%；真机套件普遍个位数任务。基准缺口 = 机会。

---

## 6. 对 LingBot-VLA 2.0 的适配建议

### 6.1 现状对照

| 维度 | 现状（本仓） | 生态位 |
|---|---|---|
| 训练数据 | RoboTwin clean 50 任务 × 50 条，LeRobot v2.1/v3.0 直读，`configs/robot_configs/<name>.yaml` 映射 + norm stats 预计算 | 单一仿真源；社区头部已是 100 万轨迹级混合 |
| 评测 | RoboTwin 2.0 50 任务（clean 93.52% / randomized 92.80%，FP32 推理）+ 开环 `scripts/open_loop_eval.py` + 真机 websocket | 只有 RoboTwin 一个公开横评点 |
| 架构 | Qwen3-VL-4B + MoE action expert + flow-matching；**55 维统一动作空间**（arm 14/eef 14/gripper 2/hand 12/waist 4/head 2/base 3/预留 4）；双查询蒸馏含 **DINO-Video 未来视频** | 动作空间设计已对齐 RDT/GR00T 路线；未来视频查询与世界模型方向天然衔接 |

### 6.2 训练数据扩充候选（按接入成本排序）

| 优先级 | 数据集 | 理由 | 接入要点 |
|---|---|---|---|
| **P0** | AgiBot World α/β | LeRobot 原生双格式；G1 本体（双臂+腰+底盘+灵巧手）与 55 维动作空间高度同构；百万轨迹级 | ⚠️ **CC BY-NC-SA 4.0 禁商用**——商用路线需单独评估；gated HF 需申请；新增 `configs/robot_configs/agibot_g1.yaml` + norm stats |
| **P0** | RoboMIND（AgileX 双臂子集） | 本仓已有 `agilex_cobot_magic.yaml`，本体直接对口；107k 轨迹多本体 | HDF5 → LeRobot 转换（社区有脚本）；gated；4 本体分别建映射 |
| **P1** | RoboMIND 2.0 | 310k 双臂 + 20k 移动操作轨迹，直接覆盖移动操作方向 | 同上，发布中 ⚠️ |
| **P1** | DROID（mini 起步） | 564 场景多样性最好；mini 300 eps 可先跑通管线 | RLDS → LeRobot 转换；单臂 Franka，动作空间需截断映射到 55 维 |
| **P1** | Mobile ALOHA / ALOHA | HF 已有 LeRobot 转换版；双臂+底盘 co-training 思路已被验证 | 直接映射；数据量小（配比权重要低） |
| **P2** | SmolVLA 社区 481 数据集 | LeRobot 原生零成本，适合数据配比/共训实验 | 低成本臂本体差异大，仅作多样性补充 |
| **P2** | OXE 子集（Bridge 等） | 生态默认共训语料 | RLDS 转换成本高；按 Octo/GR00T 配方挑选子集与采样比 |
| **P2（方向性）** | Ego4D/SSv2 视频语料 | 支撑 DINO-Video 未来视频查询与世界模型化路线（V-JEPA 2/DreamGen 配方） | 无动作数据，action-free 预训练需新增训练目标 |

### 6.3 评测基准扩展候选

| 优先级 | 基准 | 理由 | 接入要点 |
|---|---|---|---|
| **P0** | RoboTwin 2.0（保留）+ randomized 模式 | 已建成口径，clean/randomized 双模式即 OOD 泛化指标 | 已有 |
| **P1** | **SimplerEnv** | 与 RT-1/Octo/OpenVLA/CogACT/GR00T N1.7 直接同榜横评（真对齐仿真） | SAPIEN 环境；单臂 WidowX/Google Robot，需动作映射层 |
| **P1** | **RoboCasa** | 厨房家庭 100 任务、双臂友好，GR00T 系同款基准 | robosuite/MuJoCo 栈与 RoboTwin（ManiSkill3）不同，需独立环境 |
| **P1** | AgiBot World Colosseo | 100 长程任务 + G1 本体同构 + 与训练数据同生态 | 评测协议随 AgiBot-World 仓库 |
| **P2** | LIBERO | 社区横评密度最高（OpenVLA/π 复现/GR00T N1.7 均有成绩）但近饱和、单臂 | `physical-intelligence/libero` fork；仅作 backbone sanity check |
| **P2** | CALVIN ABCD→D | 长程指令链唯一标准；对照 GR-1/VPP/OpenHelix | PyBullet 栈 |
| **P2（移动方向）** | VLN-CE / HM3D + **HumanoidVLN** | 若启动 base 3 维移动能力验证；HumanoidVLN 含摔倒率，人形导航最严苛 | Habitat 栈（VLN-CE）+ Isaac Sim 栈（HumanoidVLN），均为新栈 |
| **P2（世界模型方向）** | 1X Challenge 协议 / Physics-IQ | 若将 DINO-Video 未来查询迭代为显式世界模型，可用 PSNR@77 等对标 | HF `1x-technologies/worldmodel` 数据可直接评测 |

### 6.4 注意事项

1. **License**：AgiBot World 系（CC BY-NC-SA）禁商用；RoboMIND gated；GR00T 栈 Apache 2.0 最干净；OXE 各子集不一——商用部署前必须逐子集核查。
2. **格式转换**：RLDS→LeRobot 有社区脚本但需逐数据集核对字段/相机拓扑；norm stats 必须按本仓 `scripts/compute_norm_stats.py` 口径重算（RoboTwin 先例：离线复算 2 分钟、等价官方）。
3. **动作空间映射**：55 维统一空间 → 各数据集按 `configs/robot_configs/*.yaml` 的 origin_keys 切片模式扩展；RDT（128 维物理量纲）与 GR00T N1.7（相对 EEF）的映射思路可参考；真机数据记得 `subtract_state: true`（本仓惯例）。
4. **共训配比**：Octo/GR00T/HPT 的子集采样比（非均匀）是共训效果关键变量，直接均匀混合已被多份报告证明次优。
5. **发布口径**：本项目发布验证必须 FP32 推理（BF16 有实质差异）——跨基准对比时同样注意推理精度统一，否则与 SimplerEnv/LIBERO 榜单成绩不可比。

---

## 7. 待核实清单

以下条目在调研中未能在线二次核实（多为官网/博客口径或检索失败），引用前需复核。
（注：正文所有 arXiv 编号已于 2026-09-24 经 arxiv.org 抽查确认，包括初稿中存疑的 LeRobot 2602.22818、RoboTwin-Phys 2609.26292、π0.7 2604.15483、Fleet-Scale RL 2605.00416 等，均已从本清单移除。）

| 条目 | 问题 |
|---|---|
| GR00T N1.5 | arXiv 编号未见（仅 NVIDIA blog），成绩为博客口径 |
| GR00T N2 | UGRPO 仿真轨迹规模未核实（博客无法访问） |
| TinyVLA-2 | arXiv 编号待核实 |
| InternVLA-M1 [2510.13778] | 编号已核实，但数据获取方式未确认 |
| AgiBot World β 精确规模 | α/β 合计 1,001,552 轨迹为论文口径，β 单独规模口径不一 |
| BridgeData V2 / DROID / OXE 子集 license | 「CC 系」为推断，逐子集需查 HF 卡片 |
| UniGo、RobotGO/RobotGo-2、MoVLA、NaVid-2、Butler X、BAT、Uni-Mobile、BECOME | 多源检索均未找到，疑为误记或未收录，暂不入正文 |
| AgiBot α 相机路数（7 路？） | 数字待核实（2026 版明确 9 路） |

---

*维护说明：新增基准/数据集时直接编辑本文档对应表格，并在 §7 清除已核实项。原始调研稿在 `tmp/bench_survey/`（不入库），如需溯源可重新调研或查看调研会话记录。*
