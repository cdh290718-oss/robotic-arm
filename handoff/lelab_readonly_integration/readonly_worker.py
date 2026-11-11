"""LeLab 10k BF16 observer. Imports only the validated READ-ONLY hardware adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def image_quality(images):
    import numpy as np
    result = {}
    for name, image in images.items():
        gray = image.astype(np.float32).mean(axis=2)
        result[name] = {'mean': float(gray.mean()), 'p99': float(np.percentile(gray, 99)),
                        'std': float(gray.std())}
        result[name]['near_black'] = result[name]['p99'] < 18
    return result


def run(args):
    # No imports of BoundedArm, SOFollower.connect or other write-capable adapters.
    root = args.lerobot_root
    # Running a file inside lelab/ otherwise shadows Hugging Face's `datasets`
    # package with lelab/datasets.py. Keep package imports via the editable install.
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(root / 'live_readonly_kit'))
    os.environ.update(HF_HOME=str(root/'cache/huggingface'), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import numpy as np
    from live_probe import Hardware, Engine, Preview, Replay
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    args.offline_kit = root/'offline_kit'
    args.model_dir = args.offline_kit/'models/010000/pretrained_model'
    args.robot_config = root/'data/robots/lerobot.json'
    args.calibration = root/'data/calibration/robots/so_follower/lerobot.json'
    args.port = None
    args.precision = 'bf16'
    args.gpu_memory_fraction = 0
    report = {'status': 'starting', 'mode': args.mode, 'checkpoint': '010000', 'precision': 'bf16',
              'observations': 0, 'predictions': 0, 'motor_action_execution': False,
              'created_at': time.time(), 'source_sha256': {}}
    for file in [Path(__file__), root/'live_readonly_kit/live_probe.py', root/'live_readonly_kit/readonly_bus.py']:
        report['source_sha256'][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    source = preview = engine = None
    actions = []
    states = []
    stopped = lambda: (out/'STOP').exists()
    def publish(status=None, **extra):
        if status: report['status'] = status
        report.update(extra)
        atomic_json(out/'status.json', report)
    def observe():
        row = source.snapshot()
        quality = image_quality(row['images'])
        record = {'state': row['state'].tolist(), 'camera_age_ms': row['camera_age_ms'],
                  'state_age_ms': row['state_age_ms'], 'image_quality': quality,
                  'updated_at': time.time()}
        preview.update(row, record)
        # Keep the last complete observation/prediction pair visible while the
        # next chunk is computing; otherwise the table flashes blank each tick.
        if args.mode == 'inputs' or report['predictions'] == 0:
            publish(latest=record)
        return row, record, quality
    try:
        if stopped(): raise KeyboardInterrupt()
        publish('connecting')
        source = Hardware(args, report)
        source.start()
        # Require explicit OFF state; never change a torque register to obtain it.
        if any(source.arm.torque_before.values()):
            raise RuntimeError('扭矩已开启。本入口不会改变扭矩，请先按既有流程确认机械臂状态。')
        preview = Preview(out, False)
        # Allow camera startup exposure to settle, without using a black image for inference.
        good = False
        for _ in range(20):
            if stopped(): break
            row, record, quality = observe()
            if not any(x['near_black'] for x in quality.values()):
                good = True
                break
            time.sleep(.1)
        if stopped():
            publish('stopped')
        elif not good:
            publish('input_blocked', message='画面近乎全黑，请移除两路镜头遮挡并检查照明。未加载模型。')
        else:
            if args.mode == 'live':
                publish('loading_model', message='正在核对完整 10k 权重并加载 BF16 模型。')
                engine = Engine(args, report)
                warm = Replay(args)
                for i in range(5):
                    if stopped(): break
                    publish('warming_up', warmup_completed=i)
                    engine.predict(warm.snapshot(), 20260919+i)
            start = time.perf_counter()
            publish('stopped' if stopped() else 'running', message='只读采集，不发送动作。')
            with (out/'iterations.jsonl').open('w', encoding='utf-8') as log:
                while not stopped() and time.perf_counter()-start < args.seconds:
                    tick = time.perf_counter()
                    row, record, quality = observe()
                    if any(x['near_black'] for x in quality.values()):
                        publish('input_blocked', message='检测到近乎全黑画面，已停止预测。')
                        break
                    if engine:
                        prediction, timing = engine.predict(row, 20260921+len(actions))
                        actions.append(prediction)
                        states.append(row['state'])
                        record.update(timing, first_predicted_action=prediction[0].tolist(),
                                      first_action_minus_state=(prediction[0]-row['state']).tolist())
                    record['elapsed_s'] = time.perf_counter()-start
                    preview.update(row, record)
                    log.write(json.dumps(record, allow_nan=False)+'\n');log.flush()
                    report['observations'] += 1
                    publish(latest=record, predictions=len(actions))
                    if args.mode == 'inputs':
                        time.sleep(max(0, .1-(time.perf_counter()-tick)))
                if report['status'] != 'input_blocked':
                    publish('stopped' if stopped() else 'completed')
    except KeyboardInterrupt:
        publish('stopped')
    except BaseException:
        detail = traceback.format_exc()
        publish('failed', error=detail, message=detail.strip().splitlines()[-1])
    finally:
        if source:
            try: source.close()
            except BaseException: report.setdefault('cleanup_errors', []).append(traceback.format_exc())
        if preview: preview.close()
        if actions:
            np.savez_compressed(out/'predictions.npz', actions=np.stack(actions), states=np.stack(states))
        if engine: report['gpu_memory'] = engine.memory()
        bus = report.get('read_only_bus', {})
        report['zero_writes_confirmed'] = bool(bus) and bus.get('blocked_packets') == 0 and bus.get('motor_register_write_commands_sent') == 0
        report['torque_off_confirmed'] = bool(bus.get('torque_after')) and not any(bus['torque_after'].values())
        if report.get('cleanup_errors'):
            report['status'] = 'cleanup_failed'
        report['finished_at'] = time.time()
        atomic_json(out/'summary.json', report)
        atomic_json(out/'status.json', report)
        archive = out.with_name(out.name+'_return.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in out.iterdir():
                if file.is_file() and file.name not in ('STOP','worker.log'):
                    z.write(file, file.name)
        print(json.dumps({'status': report['status'], 'archive': str(archive)}, ensure_ascii=False), flush=True)
    return 0 if report['status'] in ('completed','stopped') else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['inputs','live'], required=True)
    p.add_argument('--seconds', type=int, choices=range(1,61), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lerobot-root', type=Path, default=Path('G:/LeRobot'))
    sys.exit(run(p.parse_args()))
