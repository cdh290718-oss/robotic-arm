# 肘关节 P16 / P20 对比包

本包独立于模型和旧 Execute，不需要相机或 LeLab 网页运行。复用 `G:\LeRobot\.venv`，无需安装依赖。2026-09-21 已完成模拟测试，尚未在本机真实硬件上验证本包。此前仅验证过小角度单段 P16 诊断，不等于已验证这里的 P20 或大角度档。

## 通电后运行

关闭网页校准、遥操作和其他控制程序。机械臂起始扭矩应为关闭状态。每段结束、停止或异常时，本包会尝试关闭扭矩；请安排支撑，防止松力时下落，不要用手阻挡被测肘关节。终端保持焦点；Ctrl+C、Esc、空格可停止。软件/串口失效时不能保证成功清理，出现 cleanup_not_confirmed 必须查看错误，不能认为恢复成功。

```powershell
cd G:\LeRobot\gain_compare_kit
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-GainCompare.ps1 -Mode Check
```

Check 只读，检查训练校准一致、位置范围、扭矩关闭、位置模式、P16/I0/D32 和 Lock=1。通过显示 `STATUS: checked`，再运行小角度：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-GainCompare.ps1 -Mode Compare
```

小角度成功后，可以单独运行较大角度：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-GainCompare.ps1 -Mode Large
```

| 模式 | 肘关节目标偏移 | 每个 P 档时限 |
|---|---|---|
| Compare | 10 刻度，约 0.88° | 5 秒 |
| Large | 33 刻度，约 2.90° | 10 秒 |

每个动作模式自动执行 P16→P20 两段，采用同一组绝对位置目标：每 100ms 减 1 刻度，停留约 1 秒，再以相同步长返回起点并短暂停留。其他关节目标保持起点，夹爪固定。两段之间先关闭扭矩；第二段起始实测与第一段起点相差超过 2 刻度就退出，避免把姿态变化混入对比，不自动移动回起点。

两档跟踪误差限制均为 2°，每周期工作超过 150ms 停止。小角度实测位移限制为 2°；大角度为目标 33 刻度加 2 刻度测量余量（约 3.08°），仍低于旧 Execute 的 ±8°边界。校准范围检查保留原有 2 刻度测量余量。不是取消或扩大原有 4°跟踪保护。

本设备此前存在位置写入触发扭矩开启的现象。本包把首次目标写入明确视为可能激活电机，写入前即安排异常清理，写后确认六电机开启；不发送额外 Torque_Enable=1。若不同固件行为不符合预期则终止回传，不自动尝试其他激活方法。

仅允许肘关节 P=16/20 的精确串口配置包。不改 I/D、扭矩上限、电流保护、零点或限位，不解锁 EEPROM 来保存 P。P20 不能回读确认时退出，并尝试恢复 P16。无论正常、停止还是异常，只要尝试过 P20，都会在确认扭矩关闭后尝试恢复 P16；恢复结果明确写入报告。

## 回传什么

每次都会显示 `SEND BACK: ..._return.zip`。请回传 Check、Compare、Large 各自的 ZIP；失败包同样保留。ZIP 包含逐周期位置目标/实测、电流/负载原始值、电压/温度/状态、末段参数及扭矩回读、错误堆栈和运行源码。不需要发送权重。

如果提示 `Starting pose outside saved range`，按错误显示的关节和区间排查。此前 wrist_roll 曾为 2031、保存范围为 2042～2051。不要通过点击网页校准来查看位置或绕过错误，因为校准会改写设备零点。可以在断扭矩状态下轻微调整姿态后再次 Check；不确定时直接回传失败 ZIP。

## 不通电的代码自测

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-GainCompare.ps1 -Mode SelfTest
```

SelfTest 使用假总线，不打开 COM5。此测试成功只证明已覆盖的代码逻辑，不代表真实响应、P 写入或抓取已经验证。
