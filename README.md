# SO-101 机械臂与 SmolVLA 项目

项目包含双相机遥操作采集、黄色方块抓取数据、SmolVLA 微调，以及 Windows 离线推理、只读实时推理和实机试验的代码与记录。最新状态以 **[2026-11-12 项目状态与归因分析](docs/项目状态与归因分析_20261112.md)** 为准；先前交接文档保留历史时间点，不代表当前结论。

目前完成了 0918 数据 10k 微调、离线推理、双相机只读推理及多次固定腕部实机模型抓取尝试；机械臂能运动并碰到方块，**自主抓取放置和成功率尚未验证**。场景覆盖不足与执行时序均是有依据的排查方向，单一主因尚未确认；实际脚本和结果见[最新分析](docs/项目状态与归因分析_20261112.md)。

## 目录

- [最新状态与归因分析](docs/项目状态与归因分析_20261112.md)：近期方案、实机结果和数据/场景问题；[深夜交接](docs/项目状态与交接_20261111_深夜.md)保留8倍与V1记录。
- [历史交接 2026-09-20](docs/LeRobot实验进度与Agent交接_20260920.md)：之前部署/执行记录及来源。
- [训练和数据审查](training/)：[14组原始示教数据](datasets/ChomCUI/) 保留 0916 与 0918 的区分。
- [模型检查点](artifacts/checkpoints/)：006000 和 010000，使用 Git LFS；目前采用 010000 BF16。
- [本机运行包](handoff/windows_runtime/)：推理、只读诊断、受限动作、增益试验及原始回传。历史实机入口不是对当前硬件状态的验证。
- [离线验证脚本与样本](handoff/offline_inference/)、[设备配置与校准备份](handoff/device_configuration/)。
- [后续诊断记录](handoff/records/)、[LeLab 只读接入](handoff/lelab_readonly_integration/)、[既有实验归档](experiments/)。
- [项目原始资料和竞赛材料](references/)、[上游和本地源码快照](source/)、[归档完整性清单](handoff/manifest_20261111.json)。

## 获取大文件

模型权重及示教视频使用 Git LFS。先安装 Git LFS，再克隆并拉取：

```bash
git lfs install
git clone https://github.com/cdh290718-oss/robotic-arm.git
cd robotic-arm
git lfs pull
python tools/verify_archive.py
```

两个模型文件各约 907 MB，示教数据合计约 1.1 GiB；另有约 715 MiB 的历史训练输入 tar 包。完整克隆和校验需要足够磁盘空间与网络流量。仓库不包含 `.venv`、CUDA运行时或 Hugging Face 缓存；恢复模型推理所需的固定基础模型版本及目录映射见最新交接说明。不要把仓库代码直接覆盖正在使用的 Windows 环境。

## 当前操作边界

2026-11-11 用户在整理项目后授权了若干固定腕部实机试验；这不代表对后续所有动作的持续授权。后续训练、设备写入或实机动作应依据用户要求和最新状态；接手时先审查实际设备与最新回传。相关源码包含第三方项目，许可信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及各源码目录。
