"""Developer-machine only: export deterministic held-out RGB/state/action samples."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    repo = 'ChomCUI/yellow_cube_0918_v1'
    meta = LeRobotDatasetMetadata(repo, root=args.dataset_root)
    groups = {}
    for i, tasks in enumerate(meta.episodes['tasks']):
        groups.setdefault(tasks[0] if tasks else '', []).append(i)
    held_out = sorted(i for eps in groups.values() for i in eps[-math.ceil(len(eps)*0.15):])
    if held_out != list(range(56, 67)):
        raise ValueError(f'Unexpected held-out split: {held_out}')
    ds = LeRobotDataset(repo, root=args.dataset_root, episodes=held_out,
                       delta_timestamps={'action': [i / meta.fps for i in range(50)]},
                       video_backend='pyav', return_uint8=True)
    # Export three positions per episode, retaining all 50 future reference actions.
    ep_column = np.asarray(ds.hf_dataset['episode_index'])
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    for ep in held_out:
        indices = np.flatnonzero(ep_column == ep)
        for fraction in (0.1, 0.5, 0.85):
            idx = int(indices[min(int((len(indices)-1)*fraction), len(indices)-1)])
            item = ds[idx]
            name = f'ep{ep:03d}_frame{int(item["frame_index"]):05d}.npz'
            arrays = {k: item[k].cpu().numpy() for k in
                      ('observation.state', 'observation.images.1', 'observation.images.3',
                       'action', 'action_is_pad')}
            assert arrays['observation.images.1'].dtype == np.uint8
            assert arrays['action'].shape == (50, 6)
            np.savez_compressed(args.output / name, **arrays)
            samples.append({'file': name, 'episode_index': ep, 'frame_index': int(item['frame_index']),
                            'global_index': int(item['index']), 'task': item['task'],
                            'sha256': hashlib.sha256((args.output/name).read_bytes()).hexdigest()})
            print(name, flush=True)
    manifest = {'dataset_repo_id': repo, 'source_root': str(args.dataset_root),
                'fps': meta.fps, 'held_out_episodes': held_out, 'sample_count': len(samples),
                'sampling': '10%, 50%, 85% of each held-out episode; not full validation',
                'state_action_names': meta.features['action']['names'],
                'image_format': 'RGB CHW uint8, decoded with training PyAV backend', 'samples': samples}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
