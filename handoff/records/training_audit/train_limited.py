"""Run only this job with bounded allocator, CPU threads and step duty cycle."""
import os, time, json, subprocess
from pathlib import Path

def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.20,0)
    import lerobot.scripts.lerobot_train as trainer
    original=trainer.update_policy
    times=[]; last_check=0
    def update(*args,**kwargs):
        nonlocal last_check
        now=time.monotonic()
        if now-last_check>15:
            row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
            if int(row)<12288:
                raise RuntimeError('This job stopped: shared GPU free memory below 12 GiB; no other process modified.')
            last_check=now
        start=time.monotonic()
        result=original(*args,**kwargs)
        torch.cuda.synchronize()
        time.sleep(0.30)
        times.append(time.monotonic()-start)
        if len(times)%20==0:
            samples=times[-100:]
            stats={'steps_observed':len(times),'seconds_per_step':sum(samples)/len(samples),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'sleep_per_step':0.3}
            Path(os.environ['BENCHMARK_REPORT']).write_text(json.dumps(stats,indent=2))
            print('RESOURCE_REPORT',json.dumps(stats),flush=True)
        return result
    trainer.update_policy=update
    trainer.main()

if __name__=='__main__':main()
