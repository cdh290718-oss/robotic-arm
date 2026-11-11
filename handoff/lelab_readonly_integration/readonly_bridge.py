"""LeLab API for a separately supervised, read-only 10k BF16 worker."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict

HERE = Path(__file__).resolve().parent
ROOT = Path('G:/LeRobot')
RESULTS = ROOT/'lelab_readonly/results'
LOCK = threading.RLock()
SESSION = {}
API_LOCK = asyncio.Lock()


class StartRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['inputs','live'] = 'inputs'
    seconds: int = Field(default=20, ge=1, le=60)


def active():
    with LOCK:
        proc = SESSION.get('process')
        return proc is not None and proc.poll() is None


def read_json(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError): return {}


def status():
    with LOCK:
        if not SESSION: return {'active': False, 'status': 'idle', 'motor_action_execution': False}
        out = SESSION['output']
        result = read_json(out/'status.json')
        proc = SESSION['process']; running = proc.poll() is None
        if not result: result = {'status': 'starting' if running else 'failed'}
        if not running and not result.get('finished_at'):
            result.update(status='failed', message='工作进程已退出，未取得完整清理回读。')
        if SESSION.get('forced'):
            result.update(status='forced_stop', message='只读进程超时已终止，未确认退出时扭矩状态。',
                          torque_off_confirmed=False)
        result.update(active=running, session_id=SESSION['id'], output=str(out),
                      archive_ready=out.with_name(out.name+'_return.zip').is_file(),
                      preview_ready=(out/'latest_preview.jpg').is_file(),
                      stop_requested=(out/'STOP').exists())
        return result


def supervisor(proc, out, handle):
    started = time.monotonic(); stop_at = None
    try:
        while proc.poll() is None:
            if (out/'STOP').exists() and stop_at is None: stop_at = time.monotonic()
            if time.monotonic()-started > 180 or (stop_at is not None and time.monotonic()-stop_at > 35):
                proc.terminate()
                with LOCK:
                    if SESSION.get('process') is proc: SESSION['forced'] = True
                break
            time.sleep(.2)
        proc.wait(timeout=5)
    finally: handle.close()


def busy_reason():
    from lelab import record, teleoperate, rollout
    from lelab.calibrate import calibration_manager
    if record.recording_active: return '正在录制'
    if teleoperate.teleoperation_active: return '正在遥操作'
    rollout.handle_inference_status()
    if rollout.inference_active: return '正在实机推理'
    if calibration_manager.status.calibration_active: return '正在校准'
    return None


def start(body):
    with LOCK:
        if active(): raise HTTPException(409, '只读任务正在运行')
        reason = busy_reason()
        if reason: raise HTTPException(409, reason+'，请先停止')
        RESULTS.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out = RESULTS/run_id
        # The worker creates the session directory; log is outside it until startup.
        handle = (RESULTS/(run_id+'.log')).open('wb')
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HOME=str(ROOT/'cache/huggingface'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        cmd = [sys.executable, '-B', str(HERE/'readonly_worker.py'), '--mode', body.mode,
               '--seconds', str(body.seconds), '--output', str(out)]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                    env=env, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            handle.close();raise
        SESSION.clear();SESSION.update(process=proc, output=out, id=run_id)
        threading.Thread(target=supervisor,args=(proc,out,handle),daemon=True).start()
        return status()


def stop():
    with LOCK:
        if active():
            out = SESSION['output'];out.mkdir(parents=True,exist_ok=True)
            (out/'STOP').touch()
        return status()


def native_policy_blocked(body):
    ref = str(body.get('policy_ref','')).replace('\\','/').lower().rstrip('/')
    return '/offline_kit/models/' in ref or ref.endswith('/010000/pretrained_model') or ref.endswith('/006000/pretrained_model')


def install(app):
    router = APIRouter()
    @router.get('/so101-readonly')
    def page(): return FileResponse(HERE/'readonly.html', media_type='text/html')
    @router.get('/so101-readonly-link.js')
    def link(): return FileResponse(HERE/'readonly-link.js', media_type='text/javascript')
    @router.get('/so101-readonly/api/status')
    def get_status(): return status()
    @router.post('/so101-readonly/api/start')
    async def start_run(body: StartRequest):
        async with API_LOCK: return start(body)
    @router.post('/so101-readonly/api/stop')
    def stop_run(): return stop()
    @router.get('/so101-readonly/api/preview')
    def preview():
        with LOCK: path=SESSION.get('output',HERE/'missing')/'latest_preview.jpg'
        if not path.is_file(): raise HTTPException(404,'尚无画面')
        return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control':'no-store'})
    @router.get('/so101-readonly/api/archive')
    def archive():
        with LOCK:
            if not SESSION: raise HTTPException(404,'尚无结果')
            out=SESSION['output'];path=out.with_name(out.name+'_return.zip')
        if active() or not path.is_file(): raise HTTPException(409,'结果尚未完成')
        return FileResponse(path, filename=path.name,media_type='application/zip')
    app.include_router(router)

    # Serialize entry into native hardware workflows against read-only startup.
    starts={'/move-arm','/start-recording','/start-calibration','/start-inference','/start-port-detection','/detect-port-after-disconnect'}
    @app.middleware('http')
    async def isolate_hardware(request: Request, call_next):
        if request.url.path in starts:
            async with API_LOCK:
                if active(): return JSONResponse({'detail':'只读预测正在占用相机和串口，请先停止。'},409)
                if request.url.path == '/start-inference':
                    try: body=await request.json()
                    except ValueError: body={}
                    if native_policy_blocked(body):
                        return JSONResponse({'detail':'该模型尚未完成实机适配。请使用首页的「10k 只读预测」入口。'},409)
                return await call_next(request)
        if active() and request.url.path == '/available-cameras':
            return JSONResponse({'detail':'只读预测运行中，请勿同时扫描相机。'},409)
        return await call_next(request)

    @app.on_event('shutdown')
    async def shutdown():
        stop()
        for _ in range(100):
            if not active(): return
            await asyncio.sleep(.1)
        with LOCK:
            if active(): SESSION['process'].terminate()
