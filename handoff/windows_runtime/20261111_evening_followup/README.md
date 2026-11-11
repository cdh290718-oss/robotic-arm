# 2026-11-11 晚间实机记录

请先阅读仓库根目录的 `docs/项目状态与交接_20261111_晚间.md`。`kits` 是各次运行对应的独立代码快照；`records` 保留无图像的原始摘要、模型输出和数值日志，`.jsonl.gz` 是逐行 JSON 的 gzip 压缩。原始现场照片和可能包含照片的回传 ZIP 不在公开仓库；它们的本机路径、大小和 SHA256 见 `manifest.json` 的 `local_only_evidence`。

核验此目录（只读，不连设备）：`python verify_manifest.py`。核验仅覆盖此目录，不替代仓库根目录的 `python tools/verify_archive.py`。

模型实机入口会写入位置目标、可能触发扭矩开启，`set_d_trial*.py` 和 `lock_before_test.py` 会写设备配置。脚本是历史证据，不构成继续实机动作的指令；不要自动执行或覆盖现有环境。

`kits/elbow_start_threshold/elbow_probe_fixed.py` 是为归档中原有测试补齐的依赖，来自白天 `fixed_wrist_gain_compare`，19:22 的 `threshold.py` 没有调用它；源码逐文件来源和哈希记录在 manifest。`set_d_trial_original_1934.py` 是按抓取运行摘要的源码 SHA256 校验的历史版本，现存 `set_d_trial.py` 为之后增加写入等待的恢复版本。
