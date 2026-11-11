# SO101 双相机与状态只读验证

这一步以 **10k + BF16** 为基线，读取真实双相机与机械臂状态，记录预测动作、逐次延迟及GPU遥测。预测不会发送给机械臂。继续使用 G:\LeRobot 原环境、上一轮 offline_kit 的权重/样本及原校准，无需重下模型或安装软件。

## 1. 解压位置

将 `lerobot_live_readonly_kit.zip` 解压到 `G:\LeRobot`，得到：

```text
G:\LeRobot\live_readonly_kit\Run-LiveReadOnly.ps1
G:\LeRobot\live_readonly_kit\live_probe.py
G:\LeRobot\live_readonly_kit\readonly_bus.py
G:\LeRobot\live_readonly_kit\reference_calibration.json
```

上一轮目录保持在 `G:\LeRobot\offline_kit`。如果位置不同，可通过 `-OfflineKit` 指定；权重另存时通过 `-ModelDir` 指向 010000 的完整 pretrained_model 目录。

开始前停止 LeLab 中占用相机/COM口的采集、遥操作或推理任务。机械臂保持静止和稳妥支撑，使用原供电、原校准与相机布局。本程序保持进入时的电机扭矩状态，不开启或关闭扭矩。

## 2. 先检查输入，20秒

普通 PowerShell：

```powershell
Set-Location 'G:\LeRobot\live_readonly_kit'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-LiveReadOnly.ps1 -Mode Inputs -Seconds 20
```

此模式不加载模型。读取默认 `G:\LeRobot\data\robots\lerobot.json` 中相机1/3与 follower_port，并使用 `G:\LeRobot\data\calibration\robots\so_follower\lerobot.json`。不会扫描/重新编号设备或重做校准。

默认相机配置沿用归档逻辑：外部相机名3（此前索引1），腕部相机名1（此前索引2），两路640×480、30fps、MJPG。实际索引以你现有机器人配置为准，若USB重排请先检查画面，再修改配置中的索引；不要对调模型camera1/camera2映射。

浏览器会打开本机127.0.0.1上的预览页（临时端口），终端也打印网址。**左图应是外部视角，右图应是腕部视角。** 可以在镜头中移动物体确认画面有更新，机械臂保持静止。页中显示六维状态、时间戳和帧龄。关闭浏览器不会停止采集；20秒后自动结束，或在 PowerShell 按 Ctrl+C 提前停止。

当前环境是 opencv-python-headless，因此不调用 cv2.imshow，不安装带GUI的OpenCV。预览页面停止后不再刷新，保存的 first_preview.jpg、middle_preview.jpg、last_preview.jpg 可直接查看。预览服务只绑定本机，不对外公开。

**出现 failed、相机视角不符、校准不一致或输入过期时，先把本次返回包发回来。** 不通过重做校准来跳过检查。校准参考来自此前归档，脚本只比较文件和电机中的值，不把参考文件写进电机。

## 3. 输入正常后，只读推理60秒

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-LiveReadOnly.ps1 -Mode Live -Precision bf16 -Seconds 60
```

先从现有缓存离线加载10k模型并用旧验证样本预热5次，再连接真实输入并开始60秒采样。加载/预热不计入这60秒。状态后台目标读取30Hz，相机独立后台持续采集；每次推理取当时最新图像和状态，不等待缓存动作执行。

每次实际调用 `predict_action_chunk`，返回1×50×6动作序列，只记录到文件。脚本显式使用BF16 autocast，并保留：

```json
{"observation.images.3":"observation.images.camera1",
 "observation.images.1":"observation.images.camera2"}
```

默认最多运行60秒，Ctrl+C停止。若想保留FP16的实时对照，在相同场景、电源和性能设置下另跑：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-LiveReadOnly.ps1 -Mode Live -Precision fp16 -Seconds 60
```

实时两轮输入不完全相同，因此只能辅助比较延迟，不能直接用预测差异评价两种精度。可添加 `-NoBrowser` 关闭浏览器预览服务；仍会保存预览图片。

## 4. 可选：固定输入时延复测，无硬件连接

若需要进一步追查上次FP16延迟波动，Replay用上一轮的33个固定验证样本循环测试，逐次保存耗时及GPU状态。此模式完全不连接相机或串口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-LiveReadOnly.ps1 -Mode Replay -Precision bf16 -Seconds 60 -NoBrowser
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-LiveReadOnly.ps1 -Mode Replay -Precision fp16 -Seconds 60 -NoBrowser
```

Replay的帧龄为构造值，不是实测相机时延。这里不重复计算完整验证loss；用于观察同样输入下的时间分布及功率/频率变化。

## 5. 返回哪些结果

每次运行在 `live_readonly_kit\results` 生成独立目录和 `*_return.zip`。请把 **Inputs 和 Live 两个返回包** 发回来；如果Inputs失败，先返回Inputs包即可。

包内包括：

- `summary.json`、`SUMMARY.txt`：版本、配置哈希、状态、时延统计、相机输出帧率及只读总线记录。
- `iterations.jsonl`：每次预测的起止关联时间、预处理/模型/后处理耗时、整个循环耗时、相机帧龄、状态帧龄、双相机软件时间戳差、状态相对训练均值的偏差。
- `states.jsonl`：后台实际读取的关节状态及每次串口读取耗时；Replay没有此文件。
- `predictions.npz`：所有50×6预测动作及对应输入状态；Inputs没有此文件。
- `gpu_telemetry.csv`：约每秒一次的GPU利用率、显存、频率、功率和温度。驱动不支持的字段可为N/A，若扩展查询失败会退回基本字段并记录错误。
- `first_preview.jpg`、`middle_preview.jpg`、`last_preview.jpg`、`latest_preview.jpg`：带视角标签的现场画面；异常早停时可能只有部分图片。

回传包**包含现场相机画面和机械臂状态**，保存在本地供这次分析使用，脚本不会上传或公开这些内容。没有模型权重。脚本启动前就失败而没有ZIP时，复制PowerShell报错即可。

## 6. 只读实现与验证边界

普通 SOFollower.connect() 会调用configure并设置扭矩/寄存器；本脚本完全不走这个流程。只连接Feetech底层总线，并在串口SDK的writePort边界只放行PING、READ和SYNC_READ数据包，其他指令在发送前报错。读取也需要向串口发送查询包，“只读”指不写入电机寄存器，不是串口完全没有发送字节。

退出使用 `disconnect(disable_torque=False)`，不启用或关闭扭矩；记录前后Torque_Enable。不会写校准、工作模式、位置、速度或增益。六维状态沿用归档源码use_degrees=True：前5维为校准后的角度，夹爪为0–100归一化值。

超过250ms的相机/状态输入会使本次诊断停止并输出报告。这是排查旧帧的阈值，不是实机安全保证。相机时间戳取自软件解码返回，不是硬件曝光时间，双相机时间差不等于硬件同步精度。

模型计时排除预览编码与浏览器操作，`loop_work_ms`包含本次输入获取、推理和预览更新；浏览器/GPU遥测本身也会带来额外负载。本阶段不录制视频，因此不会修改或替代原采集系统的streaming video encoding配置。

模拟测试验证写指令拦截、校准不一致、过期输入与退出扭矩策略；开发机通过真实权重Replay验证推理与报告。**开发机没有你的相机和机械臂，Windows串口/相机实测仍由这次运行完成。**详见 `DEVELOPER_VALIDATION.json`。

源码依据：用户LeRobot归档提交da4bdcf422d97f8e5332bbae0875c5a6487ecd4d的so_follower、FeetechMotorsBus、OpenCVCamera；以及Windows已有 [feetech-servo-sdk 1.0.0](https://pypi.org/project/feetech-servo-sdk/1.0.0/) 对应的实际协议实现。脚本不覆盖这些源码。
