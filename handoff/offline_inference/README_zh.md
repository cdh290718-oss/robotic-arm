# SmolVLA Windows 离线验证工具包

用途：复用 G:\LeRobot\.venv，核对两份微调权重、基础模型缓存和真实验证样本，顺序测试 6k/10k × FP16/BF16。不会导入实机 rollout 入口、打开摄像头/串口、控制机械臂，也不会修改已安装环境或原始模型配置。

## 1. 下载和解压

推荐下载 `lerobot_offline_full.zip`，在 Windows 中解压到：

```text
G:\LeRobot\offline_kit\
  Run-Offline.ps1
  offline_eval.py
  checkpoint_manifest.json
  samples\manifest.json
  samples\*.npz
  models\006000\pretrained_model\  # 7 个文件
  models\010000\pretrained_model\  # 7 个文件
```

压缩包本身包含一层 `offline_kit`，因此解压目标选择 `G:\LeRobot`。若解压工具又增加一层文件夹，以实际 `Run-Offline.ps1` 所在目录为准。完整包约 1.7 GiB，解压需另留同等空间。基础 VLM 缓存不在此压缩包内，下一步单独检查或下载。

已有两份完整模型时可只下载约 28 MiB 的 `lerobot_offline_scripts_samples.zip`，并在调用时通过 `-ModelRoot 'G:\LeRobot\models\yellow_cube_0918_v1'` 指向包含 `006000`、`010000` 的父目录。不要指向某个单独的 pretrained_model 目录。

## 2. 先检查

在普通 PowerShell 中运行；不需要管理员权限：

```powershell
Set-Location 'G:\LeRobot\offline_kit'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Offline.ps1 -Mode Check
```

`ExecutionPolicy Bypass` 仅作用于这次新建的 PowerShell 进程，不改全局策略。所有模式显式使用 `G:\LeRobot\.venv\Scripts\python.exe`，不用系统 Python 3.14。通过 `-LeRobotRoot` 可指定其他安装位置。设置只作用于该进程及其子进程。

Check 会记录依赖、pip check、源码路径、实际 CUDA 矩阵乘法、nvidia-smi、模型和样本 SHA256，以及基础模型缓存文件是否齐全。输出 `assets_missing_or_mismatched` 不一定表示 CUDA 故障：请看 summary.json 中具体缺的是哪一项。不会自动安装、更新或降级依赖。部分 uv 环境没有安装 pip，届时 pip check 会记录“缺少 pip”，但不会因此阻止真实推理测试；请一并返回报告，不必自行升级环境。

## 3. 基础缓存不全时

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Offline.ps1 -Mode PrepareCache
```

此命令需要电脑能访问 Hugging Face，会把以下公开基础模型的必要文件下载到 `G:\LeRobot\cache\huggingface`，已有文件复用：

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
revision: 7b375e1b73b11138ff12fe22c8f2822d8fe03467
```

固定到开发机训练时实际缓存的版本，不追踪最新 main。首次下载可能约 1 GB，耗时取决于网络。下载失败则把此步骤的回传包发回来，不需要在聊天中提供令牌。PrepareCache 不下载项目的 6k/10k 权重；它们已在完整压缩包中。

## 4. 运行离线验证

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Offline.ps1 -Mode Run
```

默认运行 4 个独立进程：006000 FP16、006000 BF16、010000 FP16、010000 BF16。每个进程完整离线加载，先做一次冷启动推理，再预热 5 次、测量 100 次真实动作序列生成。失败的组合保存 traceback，后续组合继续；每个进程结束释放自己的 CUDA 上下文。终端会显示当前日志位置；100 次测试可能需要数分钟或更久，不能根据 A100 耗时预测笔记本耗时。

如先做快速检查，可以先用：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Offline.ps1 -Mode Run -Iterations 5 -Warmup 1
```

快速运行只覆盖前 5 个样本；请勿将其视作完整的 33 样本比较。正式比较仍使用默认 100 次。程序对同一个样本使用相同随机种子，每次调用 `predict_action_chunk`，不把动作队列弹出速度当成推理性能。固定种子有助于比较，但并不保证跨平台逐位一致。

## 5. 把什么返回给我

每次运行会生成独立目录，并自动打包：

```text
results\<时间>_Run\summary.json
results\<时间>_Run\SUMMARY.txt
results\<时间>_Run\006000_fp16\result.json
results\<时间>_Run\006000_fp16\console.log
results\<时间>_Run\006000_fp16\predictions.npz
... 其他组合 ...
results\<时间>_Run_return.zip
```

请发送最新的 `*_Run_return.zip`。如果在 Check 或 PrepareCache 就失败，发送对应的 `*_return.zip` 即可。包内只有诊断信息、模型/样本哈希、预测和参考动作；没有模型权重、验证图像或凭据。会包含本机解释器/源码路径和 GPU 信息。若连 Python 都无法启动，尚不能生成回传包，请直接复制 PowerShell 报错文字。

## 6. 测试含义与限制

- 样本来自 `ChomCUI/yellow_cube_0918_v1` 的验证 episode 56–66，重现训练代码按任务留出最后 ceil(67×0.15)=11 段的规则。每段抽 10%、50%、85% 处，共 33 帧；不是整套验证集。
- 两路 RGB 保持训练数据解码内容和 640×480 尺寸：外部逻辑相机3 → camera1，腕部逻辑相机1 → camera2，不伪造 camera3。脚本显式传保存的映射。
- 每帧带真实 6 维 state、未来 50 步 action 及 action_is_pad。填充的未来步不计入误差，不跨 episode 拼接动作。手臂五个关节+夹爪的原顺序不变。
- 使用 checkpoint 保存的 processor 和统计量，不重算归一化。用零/一探针检查反归一化文件重载，严格加载模型权重，不允许静默缺失参数。
- 显式 FP16 对应当前同步入口的默认 CUDA autocast 精度；显式 BF16 用于与训练精度比较。不同精度的有限值、每关节差异、动作范围都会记录。
- 动作归一化 MSE 是生成序列与记录动作的误差，不是 flow-matching 训练/验证 loss，也不是抓取成功率。33 个样本不足以决定哪个模型实机更好。
- 显存分别记录模型加载、首次推理及稳态的 PyTorch 峰值 allocated/reserved；nvidia-smi 是采样时刻的整卡用量，不是进程全程峰值。
- 延迟分别计预处理、完整 chunk 生成、后处理及整体 pipeline，边界同步 CUDA。输入已预载入 CPU；不包含读相机、读串口、视频记录和动作执行，不能据此直接声称实现 30 Hz 闭环。
- 当前阶段不生成实机运动命令。收到本地报告后，再决定精度、运行参数和后续只读实时输入检查。

## 7. 开发机证据与来源

`DEVELOPER_VALIDATION.json` 记录这份脚本在 Linux/A100 的实际验证结果与限制；它不是 Windows/RTX5060 实测结果。`export_samples.py` 是开发机导出程序，Windows 用户无需运行。每个样本在 samples/manifest.json 中保存来源 episode/frame/global index、任务文本与 SHA256。

核对的归档源码：https://github.com/cdh290718-oss/LeRobot/tree/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d

实际兼容接口：SmolVLAPolicy.from_pretrained、make_pre_post_processors、prepare_observation_for_inference、predict_action_chunk。工具包不复制或覆盖现有 LeRobot 源码。
