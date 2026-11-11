# LeLab 本地 10k 模型接入核查（2026-09-21）

已查阅 OneRobotAI/lelab 的当前上游 rollout.py、jobs.py，以及本机实际使用的 LeLab / LeRobot 源码。localhost:8000 可访问。

## 已完成

通过 LeLab 已有的 POST /jobs/import 登记本地模型，不复制、不改写权重，不启动训练或推理。

- 名称：0918 SmolVLA 10k（待适配，勿启动实机）
- job id：smolvla_imported_2026-09-21_11-44-56
- 模型目录：G:\LeRobot\offline_kit\models\010000\pretrained_model
- 检查点 API 返回本地目录正确。step=0 是导入单模型的界面占位值，不表示模型只训练了 0 步。
- 登记后 /inference-status 返回 inference_active=false。
- 目录中完整模型的七个文件均存在。本轮仅核对文件存在及尺寸，没有重新计算 906MB 主权重哈希。

这仅完成模型列表登记，未完成兼容适配或实机验收。

## LeLab 的推理方式

网页 InferenceModal 选择已登记模型、检查点、相机与任务文本，调用 /start-inference。后端 rollout.py 解析本地模型路径，再启动当前 Python 环境下的 `lerobot.scripts.lerobot_rollout` 子进程。

当前 SmolVLA 配置走默认同步推理：准备图像/状态、执行 checkpoint 预处理、policy.select_action、后处理，再发送关节目标。策略内部动作队列可缓存 chunk；它不是每个控制周期都重新做一次完整网络推理。LeLab 的自动 RTC 分支针对相对动作或 GR00T，本模型不会因为导入而自动变成 RTC。

## 必须适配的差异

| 项目 | 当前网页/标准入口 | 本项目应保持 |
|---|---|---|
| 双相机 | 遍历 config 所有 VISUAL 键，要求 camera1/2/3 全部绑定 | 仅两路：外部逻辑3→camera1；腕部逻辑1→camera2。不得复制第三路 |
| 采集尺寸 | 直接使用模型特征 shape，当前会请求 256×256 | 640×480、30FPS、MJPG，模型输入变换交给原 processor |
| 映射 | LeLab 未传 rename_map；rollout 以默认空映射覆盖保存映射 | 显式保留训练映射，或者完整验证等价输入命名方案 |
| 精度 | sync.py 的 autocast 没有显式 dtype | 明确 torch.bfloat16；use_amp 不等于 BF16 |
| 单位 | 本机 SOFollowerConfig.use_degrees=True | 已一致；前五维角度，夹爪0–100，不需重校准 |
| 启动 | robot.connect→configure：写模式、PID、夹爪限矩等并开启扭矩 | 保留既有配置，先处理目标写入后扭矩变化和首次写入异常清理 |
| 动作保护 | max_relative_target 默认 None；无本项目整套10秒/±8°/单tick/4°跟随保护 | 不能以更换入口为由跳过保护 |
| 停止 | 正常 teardown 默认可返回起始位置；网页 stop 使用 terminate | 应明确保持/释放语义，Windows 强制结束不能假定 Python finally 会执行 |
| 校准 | 不一致时可能写回已有校准；LeLab 给子进程输入换行 | 本项目应只校验一致性，不自动写回校准 |

核心依据（本机）：

- G:\LeRobot\source\lelab-9a182993a2a9a3b04312c13ff8365f68d379fe80\lelab\rollout.py
- 同目录 lelab\jobs.py、frontend\src\components\landing\InferenceModal.tsx
- G:\LeRobot\source\lerobot-0.6.0\src\lerobot\rollout\context.py
- 同目录 rollout\inference\sync.py、robots\so_follower\so_follower.py、robots\so_follower\config_so_follower.py

## 最新失败包

20260921_113458_841_Execute_bf16_return.zip CRC 通过，控制周期为0。启动对齐后 Present_Position 同步读取报告 There is no status packet。全程只有一次位置写入；起始扭矩全0，退出全1，stop_hold为空。故这是启动通信/清理问题，不是本次运行中的4°跟随误差，也不是模型推理异常。

20260921_113801_614_Release_bf16_return.zip CRC 通过，扭矩从全1回读为全0。这是该包采集时状态，本轮未再操作硬件。

## 接入结论

模型格式和本地导入功能兼容，无需重新训练。可复用 LeLab 的模型管理、相机预览、状态与日志界面，但目前不能直接把“已登记”当成“可以点击开始实机”。应先接入零写入的预测验证入口，确认镜头无遮挡及输入正确，再整合经过审查的执行与停止逻辑。此次保留旧源代码和执行包原样，未部署上述适配，也未调用 /start-inference。

参考上游：https://github.com/OneRobotAI/lelab ，https://github.com/OneRobotAI/lelab/blob/main/lelab/rollout.py 。
