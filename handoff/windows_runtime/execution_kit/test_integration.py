"""Exercise production main/cleanup with fake serial/cameras and a spawned predictor."""
import contextlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import bounded_run as runner
from control_core import BoundedArm, MOTORS
from test_control import FakeBus, CAL


def fake_predictor(args,shared,results,stop):
    results.cancel_join_thread()
    results.put({'kind':'ready'})
    seq=0
    while not stop.is_set():
        row=shared.get(seq)
        if row is None:
            time.sleep(.005);continue
        seq=row['sequence']
        action=np.tile(row['state'],(50,1));action[:,0]+=5
        msg={'kind':'chunk','id':seq,'observation_time':min(row['state_timestamp'],*row['image_timestamps'].values()),
             'actions':action.tolist()}
        try:results.put(msg,timeout=.01)
        except Exception:pass
        time.sleep(.02)


class FakeRig:
    def __init__(self,*args):
        self.images={k:np.zeros((480,640,3),np.uint8) for k in ('3','1')}
    def start(self):pass
    def row(self,state,stamp):
        return {'images':self.images,'state':state,'state_timestamp':stamp,'image_timestamps':{'3':stamp,'1':stamp}}
    def close(self):return []


class FakeTelemetry:
    def __init__(self,*args):self.info={}
    def start(self):pass
    def close(self):pass


class FollowingBus(FakeBus):
    def __init__(self,fail=False):
        super().__init__();self.goals=0;self.fail=fail
    def sync_write(self,name,values,**kwargs):
        if name=='Goal_Position':
            self.goals+=1
            if self.fail and self.goals==4:
                raise ConnectionError('Injected one-time serial fault')
        super().sync_write(name,values,**kwargs)
        if name=='Goal_Position':self.data['Present_Position']=self.data[name].copy()


class TestMain(unittest.TestCase):
    def run_case(self,mode,fail=False):
        bus=FollowingBus(fail)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);config=tmp/'config.json';cal=tmp/'cal.json';out=tmp/'result'
            config.write_text(json.dumps({'follower_port':'fake','cameras':[]}))
            cal.write_text(json.dumps(CAL))
            args=['bounded_run.py','--mode',mode,'--seconds','.30','--output',str(out),
                  '--robot-config',str(config),'--calibration',str(cal)]
            def make_arm(port,calibration,writable):
                return BoundedArm(port,calibration,writable,lambda:bus)
            previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                with patch.object(sys,'argv',args), patch.object(runner,'BoundedArm',make_arm), \
                     patch.object(runner,'CameraRig',FakeRig), patch.object(runner,'Telemetry',FakeTelemetry), \
                     patch.object(runner,'inference_worker',fake_predictor), \
                     patch.object(runner,'save_panel'), contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    code=runner.main()
            finally:
                for s,handler in previous.items():signal.signal(s,handler)
            summary=json.loads((out/'summary.json').read_text())
            import zipfile
            with zipfile.ZipFile(tmp/'result_return.zip') as z:self.assertIsNone(z.testzip())
        return code,summary,bus
    def test_execute_completion_holds(self):
        code,r,bus=self.run_case('execute')
        self.assertEqual(code,0,r)
        self.assertEqual(r['status'],'completed')
        self.assertGreater(r['control_ticks'],0)
        self.assertTrue(r['stop_hold']['goal_readback_confirmed'])
        self.assertEqual(bus.data['Torque_Enable'],[1]*6)
    def test_fault_holds_before_teardown(self):
        code,r,bus=self.run_case('execute',True)
        self.assertEqual(code,1)
        self.assertEqual(r['status'],'failed')
        self.assertTrue(r['stop_hold']['goal_readback_confirmed'])
        self.assertIn('Injected one-time serial fault',r['error'])
        self.assertLess(len(bus.events)-1-bus.events[::-1].index('write_Goal_Position'),bus.events.index('disconnect_without_torque_change'))
    def test_dryrun_never_writes(self):
        code,r,bus=self.run_case('dryrun')
        self.assertEqual(code,0,r)
        self.assertEqual(r['bus']['write_attempts'],0)
        self.assertFalse(any(e.startswith('write_') for e in bus.events))
    def test_release_skips_model_and_cameras(self):
        code,r,bus=self.run_case('release')
        self.assertEqual(code,0,r)
        self.assertEqual(r['status'],'released')
        self.assertNotIn('worker_exitcode',r)

if __name__=='__main__':
    mp.freeze_support()
    unittest.main(verbosity=2)
