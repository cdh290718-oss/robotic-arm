# LeRobot：SO-ARM101 VLA 项目

基于 SO-ARM101 主从机械臂、双 RGB 摄像头、LeLab / LeRobot 和 SmolVLA 的桌面黄方块抓取验证项目。

**最新状态（2026-09-20）：10k 微调、本机离线/只读实时推理及两次10秒受限实机执行已完成。肘关节跟随不足和启动扭矩异常仍待排查；旧 Execute 暂停使用，下一步为只读电机诊断。尚未验证完整抓取成功率。**

- **下一位 Agent 从这里开始：[最新实验进度与 Agent 交接](docs/LeRobot实验进度与Agent交接_20260920.md)。**
- [本地和开发机目录、恢复方式](docs/目录与恢复说明.md)
- [数据集清单](docs/dataset_inventory.json) / [训练来源到合并 episode 的映射](training/episode_provenance.json)
- [实测训练记录](training/remote-records/logs/train10k.summary.log)
- [用户提供的资料索引](references/资料索引.md)
- [第三方源码来源与许可](THIRD_PARTY_NOTICES.md)

## 仓库结构

| 目录 | 内容 |
|---|---|
| `source/` | 当前本地使用的 LeLab 和 LeRobot 源码快照，包含本地修改与原有许可 |
| `local/` | Windows 启停脚本、G盘环境配置、校准和设备配置、安装验收记录 |
| `training/` | 数据审核、合并、限资源训练脚本、200步测速和10k训练记录 |
| `docs/` | 最新交接说明、目录地图、完整归档清单 |
| `references/` | 项目原始 Word/PDF、竞赛材料、商家参数和故障截图 |
| `experiments/` | 0919部署、验证、受限执行及只读诊断代码与报告，含33个离线验证样本 |
| `cui_local_computer_files/results/` | 本机原始回传ZIP，供复核 |
| `archive/` | 旧排障工具、旧数据清单、修改前备份；不是当前状态依据 |

包含代码、配置、报告、33个离线验证样本及小型原始回传。**不包含模型权重、训练数据视频、Python环境、缓存、令牌或SSH私钥。** 原始运行目录仍在 `G:\LeRobot`，本归档不会替换正在使用的安装。

仓库中的目录和校准信息对应当前设备；克隆到其他电脑后不能直接假设 COM 端口、相机索引和 G 盘路径相同。内置源码的部分上游测试大文件已排除，完整上游测试需要另行获取其测试资源。

`training/训练方案.md`、`local/使用说明.md` 等保留历史版本以便追溯；当前进度以 `docs/LeRobot实验进度与Agent交接_20260920.md` 为准；0918交接保留为历史记录。
