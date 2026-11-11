"""SO101 SmolVLA offline diagnostics. Never opens cameras, serial ports, or robots.
Uses one subprocess per checkpoint/AMP mode to release CUDA memory completely.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
BASE_MODEL = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
BASE_REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
RENAME = {'observation.images.3': 'observation.images.camera1',
          'observation.images.1': 'observation.images.camera2'}
JOINTS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, errors='replace', timeout=90)
        return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except Exception as exc:
        return {'error': str(exc)}


def audit_environment():
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'hf_home': os.environ.get('HF_HOME'), 'packages': {}}
    for name in ('torch', 'torchvision', 'torchcodec', 'lerobot', 'transformers', 'numpy',
                 'safetensors', 'huggingface_hub', 'av', 'accelerate'):
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = 'MISSING'
    result['pip_check'] = command_output([sys.executable, '-m', 'pip', 'check'])
    result['nvidia_smi'] = command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free', '--format=csv'])
    try:
        import torch
        import lerobot
        result.update(lerobot_source=list(lerobot.__path__), cuda_available=torch.cuda.is_available(),
                      cuda_runtime=torch.version.cuda)
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name(0)
            result['capability'] = list(torch.cuda.get_device_capability(0))
            result['bf16_supported'] = torch.cuda.is_bf16_supported()
            # Exercise an actual CUDA kernel, not only driver enumeration.
            x = torch.arange(256, device='cuda', dtype=torch.float32).reshape(16, 16)
            result['cuda_kernel_ok'] = bool(torch.isfinite(x @ x.T).all().item())
            torch.cuda.synchronize()
    except Exception:
        result['import_or_cuda_error'] = traceback.format_exc()
    return result


def audit_assets(args):
    manifest = json.loads((HERE / 'checkpoint_manifest.json').read_text(encoding='utf-8'))
    result = {'checkpoints': {}, 'samples': {}, 'base_cache': {}}
    for step in args.checkpoints:
        folder = args.model_root / step / 'pretrained_model'
        errors = []
        files = {}
        for name, expected in manifest['checkpoints'][step]['files'].items():
            path = folder / name
            info = {'exists': path.is_file()}
            if path.is_file():
                info.update(bytes=path.stat().st_size, sha256=sha256(path))
                if info['bytes'] != expected['bytes'] or info['sha256'] != expected['sha256']:
                    errors.append(f'Hash/size mismatch: {name}')
            else:
                errors.append(f'Missing: {name}')
            files[name] = info
        if not errors:
            processor = json.loads((folder/'policy_preprocessor.json').read_text(encoding='utf-8'))
            maps = [s['config']['rename_map'] for s in processor['steps'] if s.get('registry_name') == 'rename_observations_processor']
            if maps != [RENAME]:
                errors.append('Saved camera mapping differs from training mapping')
        result['checkpoints'][step] = {'path': str(folder), 'ok': not errors, 'errors': errors, 'files': files}
    sample_meta = json.loads((args.samples/'manifest.json').read_text(encoding='utf-8'))
    errors = []
    for item in sample_meta['samples']:
        path = args.samples/item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            errors.append(item['file'])
    if sample_meta['held_out_episodes'] != list(range(56, 67)) or sample_meta['state_action_names'] != JOINTS:
        errors.append('Unexpected validation split or joint order')
    result['samples'] = {'ok': not errors, 'errors': errors, 'count': len(sample_meta['samples'])}
    try:
        from huggingface_hub import snapshot_download
        cache_path = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True))
        missing = [name for name in manifest['base_cache_files'] if not (cache_path/name).is_file()]
        result['base_cache'] = {'path': str(cache_path), 'revision': BASE_REVISION, 'missing_files': missing,
                                'ok': not missing, 'note': 'File existence only; actual offline loading is tested by Run.'}
    except Exception:
        result['base_cache'] = {'ok': False, 'error': traceback.format_exc()}
    result['ok'] = (all(x['ok'] for x in result['checkpoints'].values())
                    and result['samples']['ok'] and result['base_cache']['ok'])
    return result


def worker(args):
    # Set before importing torch/transformers/HF; enforce a real offline load.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this interpreter')
    if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unsupported on this GPU/build')
    if args.gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    folder = args.model_root / args.checkpoint / 'pretrained_model'
    sample_meta = json.loads((args.samples/'manifest.json').read_text(encoding='utf-8'))
    samples = []
    for item in sample_meta['samples']:
        with np.load(args.samples/item['file'], allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        if arrays['observation.state'].shape != (6,) or arrays['action'].shape != (50, 6):
            raise ValueError('Unexpected state/action dimensions')
        for camera in ('1', '3'):
            img = arrays[f'observation.images.{camera}']
            if img.dtype != np.uint8 or img.shape != (3, 480, 640):
                raise ValueError(f'Unexpected RGB shape or dtype: {img.shape}, {img.dtype}')
        if not np.isfinite(arrays['observation.state']).all() or not np.isfinite(arrays['action']).all():
            raise ValueError('Nonfinite reference data')
        samples.append((item, arrays))
    snapshot = snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    cfg = PreTrainedConfig.from_pretrained(str(folder), local_files_only=True)
    cfg.device = 'cuda'
    cfg.pretrained_path = folder
    cfg.compile_model = False
    # Pin the SAME base assets used during training, without modifying saved configs.
    cfg.vlm_model_name = snapshot
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(str(folder), config=cfg, local_files_only=True, strict=True)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(folder),
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                               'rename_observations_processor': {'rename_map': RENAME},
                               'tokenizer_processor': {'tokenizer_name': snapshot}})
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    def memory():
        free, total = torch.cuda.mem_get_info()
        return {'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                'current_allocated_mib': torch.cuda.memory_allocated()/2**20,
                'device_free_mib': free/2**20, 'device_total_mib': total/2**20}
    load_memory = memory()
    stats = load_file(str(folder/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    std_key = next(k for k in stats if 'action' in k and k.endswith('std'))
    std = stats[std_key].float().cpu().numpy().reshape(6)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid action normalizer standard deviation')
    mean_key = next(k for k in stats if 'action' in k and k.endswith('mean'))
    mean = stats[mean_key].float().cpu().numpy().reshape(6)
    with torch.inference_mode():
        probes = torch.stack([torch.zeros(6), torch.ones(6)]).unsqueeze(0).cuda()
        restored = post(probes).float().cpu().numpy()[0]
    np.testing.assert_allclose(restored[0], mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(restored[1], mean + std, rtol=1e-5, atol=1e-5)
    post.reset()

    def predict(index):
        item, data = samples[index]
        policy.reset(); pre.reset(); post.reset()
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs = {'observation.state': data['observation.state'].copy()}
        for cam in ('1', '3'):
            obs[f'observation.images.{cam}'] = data[f'observation.images.{cam}'].transpose(1, 2, 0).copy()
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            batch = prepare_observation_for_inference(obs, torch.device('cuda'), item['task'], 'so101_follower')
            batch = pre(batch)
            expected_images = {'observation.images.camera1', 'observation.images.camera2'}
            if {k for k in batch if k.startswith('observation.images.')} != expected_images:
                raise ValueError(f'Unexpected camera keys: {list(batch)}')
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            normalized = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            actions = post(normalized).float().cpu()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
        if tuple(actions.shape) != (1, 50, 6):
            raise ValueError(f'Unexpected action shape {actions.shape}')
        if not bool(torch.isfinite(normalized).all()) or not bool(torch.isfinite(actions).all()):
            raise ValueError(f'Nonfinite action in {args.amp}, sample {index}')
        return actions[0].numpy(), {'pre_ms': (t1-t0)*1000, 'chunk_ms': (t2-t1)*1000,
                                   'post_ms': (t3-t2)*1000, 'pipeline_ms': (t3-t0)*1000}

    cold_pred, cold_timing = predict(0)
    cold_memory = memory()
    for i in range(args.warmup):
        predict(i % len(samples))
    torch.cuda.reset_peak_memory_stats()
    timings = []
    predictions = {}
    for i in range(args.iterations):
        index = i % len(samples)
        pred, timing = predict(index)
        timings.append(timing)
        predictions.setdefault(index, pred)
        if (i+1) % 10 == 0 or i == 0:
            print(f'{args.checkpoint} {args.amp}: {i+1}/{args.iterations} real chunks', flush=True)
    steady_memory = memory()
    ids = sorted(predictions)
    predicted = np.stack([predictions[i] for i in ids])
    target = np.stack([samples[i][1]['action'] for i in ids])
    valid = np.stack([~samples[i][1]['action_is_pad'].astype(bool) for i in ids])
    errors = (predicted-target)[valid]
    if not len(errors):
        raise ValueError('No valid reference actions')
    np.savez_compressed(args.output/'predictions.npz', predicted=predicted, reference=target,
                        valid=valid, sample_indices=np.asarray(ids), first_cold_prediction=cold_pred)
    latency = {key: {'p50': float(np.percentile([t[key] for t in timings], 50)),
                     'p95': float(np.percentile([t[key] for t in timings], 95)),
                     'mean': float(np.mean([t[key] for t in timings]))} for key in timings[0]}
    return {'status': 'passed', 'checkpoint': args.checkpoint, 'amp': args.amp,
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
            'load_seconds': load_seconds, 'load_memory': load_memory,
            'cold_first_chunk_ms': cold_timing, 'load_and_cold_memory': cold_memory,
            'steady_memory': steady_memory, 'warmup': args.warmup, 'iterations': args.iterations,
            'seed': args.seed, 'latency_ms': latency, 'saved_unnormalizer_probe_passed': True,
            'sample_count': len(ids), 'sample_indices': ids, 'valid_reference_steps': int(valid.sum()),
            'held_out_normalized_action_mse': float(np.mean((errors/(std+1e-8))**2)),
            'held_out_mae_per_joint': dict(zip(JOINTS, np.abs(errors).mean(axis=0).tolist())),
            'prediction_min_per_joint': predicted.min(axis=(0, 1)).tolist(),
            'prediction_max_per_joint': predicted.max(axis=(0, 1)).tolist(),
            'all_actions_finite': True, 'chunk_shape': [1, 50, 6],
            'nvidia_smi_after': command_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv']),
            'limits': 'Offline 33-frame subset, not full validation loss or grasp success; timing excludes camera/serial/video I/O. Same seed per sample, each measured call generates a fresh full chunk.'}


def controller(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'status': 'started',
              'robot_connected': False, 'runs': [], 'arguments': {k: str(v) for k, v in vars(args).items()}}
    try:
        if args.mode == 'cache':
            from huggingface_hub import snapshot_download
            manifest = json.loads((HERE/'checkpoint_manifest.json').read_text(encoding='utf-8'))
            # Explicit command only: fetch the pinned original base model assets.
            report['base_cache'] = snapshot_download(BASE_MODEL, revision=BASE_REVISION,
                                                     allow_patterns=manifest['base_cache_files'])
            report['status'] = 'cache_prepared'
        else:
            # A separate probe process releases its CUDA context before inference.
            probe = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                    '--mode', 'environment', '--output', str(args.output)],
                                   capture_output=True, text=True, errors='replace')
            (args.output/'environment.log').write_text(probe.stdout + probe.stderr, encoding='utf-8')
            report['environment'] = json.loads((args.output/'environment.json').read_text(encoding='utf-8'))
            report['assets'] = audit_assets(args)
            write_json(args.output/'summary.json', report)
            if args.mode == 'check':
                report['status'] = 'check_complete' if report['assets']['ok'] else 'assets_missing_or_mismatched'
            elif not report['assets']['ok']:
                report['status'] = 'blocked_assets'
            elif not report['environment'].get('cuda_kernel_ok'):
                report['status'] = 'blocked_cuda'
            else:
                for step in args.checkpoints:
                    for amp in args.precisions:
                        folder = args.output / f'{step}_{amp}'
                        folder.mkdir()
                        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', 'worker',
                               '--checkpoint', step, '--amp', amp, '--model-root', str(args.model_root),
                               '--samples', str(args.samples), '--output', str(folder),
                               '--iterations', str(args.iterations), '--warmup', str(args.warmup),
                               '--seed', str(args.seed), '--gpu-memory-fraction', str(args.gpu_memory_fraction)]
                        print(f'Running {step} {amp}; log: {folder / "console.log"}', flush=True)
                        with (folder/'console.log').open('w', encoding='utf-8') as log:
                            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                        worker_file = folder/'result.json'
                        entry = json.loads(worker_file.read_text(encoding='utf-8')) if worker_file.exists() else {'status': 'process_failed'}
                        entry.update(returncode=r.returncode, checkpoint=step, amp=amp, folder=folder.name)
                        report['runs'].append(entry)
                        write_json(args.output/'summary.json', report)
                report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['runs']) else 'partial_or_failed'
                # Compare precision effects on the same held-out inputs and random seeds.
                import numpy as np
                comparisons = {}
                for step in args.checkpoints:
                    files = [args.output/f'{step}_{amp}'/'predictions.npz' for amp in ('fp16', 'bf16')]
                    if all(f.exists() for f in files):
                        with np.load(files[0]) as a, np.load(files[1]) as b:
                            if np.array_equal(a['sample_indices'], b['sample_indices']):
                                delta = np.abs(a['predicted']-b['predicted'])
                                comparisons[step] = {'mean_abs_per_joint': delta.mean(axis=(0,1)).tolist(),
                                                     'max_abs_per_joint': delta.max(axis=(0,1)).tolist()}
                report['fp16_vs_bf16'] = comparisons
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        write_json(args.output/'summary.json', report)
        lines = [f"Status: {report['status']}", 'Offline only; no robot/camera/serial connection.',
                 'Details: summary.json. Report is NOT a real-robot deployment approval.', '']
        for r in report['runs']:
            lines.append(f"{r['checkpoint']} {r['amp']}: {r['status']}")
            if r['status'] == 'passed':
                lines.append(f"  chunk P50/P95 ms: {r['latency_ms']['chunk_ms']['p50']:.2f}/{r['latency_ms']['chunk_ms']['p95']:.2f}")
                lines.append(f"  pipeline P50/P95 ms: {r['latency_ms']['pipeline_ms']['p50']:.2f}/{r['latency_ms']['pipeline_ms']['p95']:.2f}")
                lines.append(f"  held-out normalized action MSE: {r['held_out_normalized_action_mse']:.6f}")
        (args.output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        archive = args.output.with_name(args.output.name+'_return.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(args.output.rglob('*')):
                if file.is_file():
                    z.write(file, file.relative_to(args.output))
        print('\n'.join(lines), flush=True)
        print(f'SEND BACK: {archive}', flush=True)
    return 0 if report['status'] in ('passed', 'check_complete', 'cache_prepared') else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['check', 'cache', 'run', 'worker', 'environment'], default='check')
    p.add_argument('--model-root', type=Path, default=HERE/'models')
    p.add_argument('--samples', type=Path, default=HERE/'samples')
    p.add_argument('--output', type=Path, default=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--checkpoints', nargs='+', choices=['006000','010000'], default=['006000','010000'])
    p.add_argument('--precisions', nargs='+', choices=['fp16','bf16'], default=['fp16','bf16'])
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--seed', type=int, default=20260919)
    p.add_argument('--checkpoint', choices=['006000','010000'])
    p.add_argument('--amp', choices=['fp16','bf16'])
    p.add_argument('--gpu-memory-fraction', type=float, default=0.0,
                   help='Optional allocator cap for developer-machine smoke tests; 0 means no cap')
    args = p.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        p.error('iterations must be positive and warmup nonnegative')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if args.mode != 'cache':
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    else:
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
    if args.mode == 'environment':
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'environment.json', audit_environment())
        return 0
    if args.mode == 'worker':
        args.output.mkdir(parents=True, exist_ok=True)
        try:
            result = worker(args)
        except Exception:
            result = {'status': 'failed', 'error': traceback.format_exc()}
            print(result['error'], flush=True)
        write_json(args.output/'result.json', result)
        return 0 if result['status'] == 'passed' else 1
    return controller(args)

if __name__ == '__main__':
    sys.exit(main())
