# SmolVLA 10k：Windows 首轮受限执行包

复用 `G:\LeRobot\.venv`、现有双相机/COM5 标定和 `offline_kit` 中的 10k 权重。默认 BF16，不安装依赖，不下载模型，不改标定。本包首次加入实际电机命令；`DryRun` 默认只记录，`Execute` 才会动作。

本轮是 **最多 10 秒的小幅跟随与停止测试，夹爪固定**，还不是完整抓取评估。不要同时运行旧的 Inputs/Live、LeLab 控制程序或其他串口/相机进程。

## 1. 下载与解压

将 `lerobot_bounded_execution_kit.zip` 下载到本机，解压到 `G:\LeRobot`，应得到：

```text
G:\LeRobot\execution_kit\Run-Bounded.ps1
G:\LeRobot\execution_kit\bounded_run.py
G:\LeRobot\execution_kit\control_core.py
```

已有模型应保留在：

```text
G:\LeRobot\offline_kit\models\010000\pretrained_model\
G:\LeRobot\cache\huggingface\
```

如果模型路径不同，在命令末尾加 `-OfflineKit '你的原 offline_kit 目录'` 或 `-ModelDir '包含 model.safetensors 的 pretrained_model 目录'`。本包没有重复打包权重。

## 2. 先做本机不发送动作的演练

保持此前相机布局和任务场景，机械臂处于稳定、有支撑的位置，电源开启以便读状态。使用原环境，无需另建 Python 3.14 环境。

```powershell
cd G:\LeRobot\execution_kit
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Bounded.ps1 -Mode DryRun -Seconds 10
```

加载和五次模型预热不计入 10 秒；首次加载可能需要一两分钟。DryRun 会计算完整模型动作及限幅后的候选命令，**不会写目标位置或扭矩**。等待终端显示 `STATUS: completed`。失败时把回传 ZIP 发回来，先不要继续 Execute。

此版本不打开浏览器窗口，避免图像编码影响控制循环。输出包含开始及末次输入的双相机图：左侧外部相机 3，右侧腕部相机 1。

## 3. 首轮实际动作

仅在 DryRun 完成、机械臂周围有小幅运动空间且你在机器旁可停止时运行。初始要求六个电机扭矩全为 0；手不要放在夹爪或运动路径中。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Bounded.ps1 -Mode Execute -Seconds 10
```

脚本先将六个电机的目标位置写为**刚读到的当前位置**，读回确认目标一致，再启用扭矩。启用前检查位置模式、现有校准和位置稳定性，不调用 SOFollower 的自动配置或重新标定。

首轮约束固定在代码中，未提供放宽参数：

| 项目 | 约束 |
|---|---|
| 执行时间 | 最多 10 秒，时长含末次循环开销，不是硬实时截止 |
| 身体关节目标 | 相对启动位置最多 ±8°，同时落在已有标定范围内 |
| 目标变化速度 | 上限 5°/秒；按原始编码器整数向下取整后，每次最多 1 tick，30 Hz 时约 2.64°/秒 |
| 夹爪 | 固定启动位置，不能做夹取/松开 |
| 追踪误差 | 实测与上一目标偏差超过 4°时停止推进 |
| 循环卡顿 | 单次间隔或写命令前的工作超过 150 ms 时停止推进 |
| 图像/状态 | 输入年龄大于 250 ms 时停止 |
| 新预测 | 观察年龄大于 1.4 秒的结果不启用 |
| 动作序列耗尽 | 保持最后目标；连续 2 秒无可用序列则结束并尝试当前位置保持 |

关节速度约束针对**发送的目标值**，不是实测速度或碰撞保证。原标定 `wrist_roll` 仅为 2042–2051，共 9 tick（约 0.79°）。本包保留它并保守限幅；可能几乎看不到该关节运动。其他接近标定边界的关节也可能被限制，回传数据会逐关节记录。

模型每次仍输出 50 个动作，使用训练的 30 Hz 时间基准。收到结果时跳过已过时的前若干步，再对当前动作限幅。这是简单时间对齐，不是完整 RTC，也不代表动作准确性已验证。

## 4. 停止及解除扭矩

让 PowerShell 窗口保持焦点。`Ctrl+C`、`Esc`、空格或 `Q` 都请求停止；也会在 10 秒结束后自动停止。停止处理先读当前位置，再将该位置写作保持目标并读回核对，**默认保留扭矩开启，机械臂仍通电保持**。

需要解除扭矩时，先在本地承托好机械臂、避免其下落，再运行独立命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Bounded.ps1 -Mode Release
```

Release 不加载模型、不打开相机，只将六个电机扭矩关闭并核对回读。下一次 Execute 前需要回到稳定支撑、扭矩关闭的起点。

停止依赖正常的 Windows 进程和串口。USB 断开、进程被强制杀死或电机通信失效时，脚本不能保证保持命令到达；现场需要能物理停止设备。不要把关闭终端当成已确认停止。若输出 `hold_not_confirmed`，先现场处置再回传日志。

## 5. 把这些结果发回来

脚本结束会显示 `SEND BACK: ..._return.zip`。回传 `DryRun` 和 `Execute` 两个 ZIP；如果使用 Release，也附上它的 ZIP。失败的 ZIP 同样保留。

```text
execution_kit\results\日期时间_DryRun_bf16_return.zip
execution_kit\results\日期时间_Execute_bf16_return.zip
```

包内主要内容：

- `summary.json`：实际模式、起点、边界、写包计数、停止原因和保持回读。
- `control.jsonl`：每个周期的实测位置、原始预测、实际目标、限幅标志和输入年龄。
- `chunks.jsonl`、`predictions.npz`：完整 50 步预测及其时间信息。
- `inference_summary.json`、`inference.log`：模型加载、显存和推理异常。
- `camera_before.jpg`、`camera_last_observation.jpg`：开始/末次观察，两者都不是实机成功证据。
- `gpu_telemetry.csv`：GPU 采样。

如果方便，附上手机录制的短视频，覆盖启动、移动及停止；仅日志不能判断桌面碰撞或实际抓取效果。

## 实现和验证边界

`control_core.py` 是动作边界和串口门控；串口底层仅允许读包，以及一次显式授权的完整 Goal_Position / Torque_Enable 同步写包，其他写入被阻止。`bounded_run.py` 的主进程独占串口，模型在独立 spawn 进程中运行，使用单份共享内存传递最新图像和状态，不积压帧队列。

`live_probe.py` / `readonly_bus.py` 是此前已验证代码的原样副本，供复用模型加载和标定检查；本包入口是 `Run-Bounded.ps1`。模型七个文件和基础模型缓存仍核对原始固定版本。

开发机已进行实际 Feetech SDK 包门控、完整故障清理测试和真实 10k BF16 模型的模拟控制联调。详细结果见 `DEVELOPER_VALIDATION.json`。**模拟执行没有连接机械臂，不能验证真实电机动力学、Windows 控制延迟或抓取效果；这些由这次本机回传确认。**
