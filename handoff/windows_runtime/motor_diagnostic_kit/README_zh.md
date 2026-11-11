# 电机只读诊断：return1 后续

先暂停旧执行包的 Execute。此包不加载模型、不打开相机，不写位置、扭矩、PID、校准或速度参数；它保留机器当前扭矩状态，不会自动释放。

解压到 `G:\LeRobot` 得到 `motor_diagnostic_kit`。关闭其他机械臂控制程序，在机械臂处于稳定支撑的情况下运行；无需手动摆动或尝试让肘关节运动。

```powershell
cd G:\LeRobot\motor_diagnostic_kit
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-MotorDiagnostic.ps1
```

约采集 15 秒；Ctrl+C 可结束。把终端显示的 `_MotorDiagnostic_return.zip` 回传。即使出现读取错误，也回传完整包。

记录各关节位置/目标、扭矩、负载、电流、电压、温度、状态字，以及已有模式、PID、限矩和速度参数。所有数值保留 SDK 原始值，不擅自按不明硬件版本换算电流/电压。各寄存器顺序读取，并非同一时刻的快照。

本包只能帮助缩小问题范围，不能复现之前启动时的扭矩变化，也不能用静止诊断排除运动时堵转或供电不足。不要为了复现问题继续运行旧 Execute 或增大误差阈值。

串口写入口复用已验证的 `ReadOnlyPacketGuard`，仅允许 PING/READ/SYNC_READ 协议包。电脑实际电机诊断尚待本次回传；无硬件的正常和通信失败路径已测试。
