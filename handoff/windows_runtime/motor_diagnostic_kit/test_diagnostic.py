import contextlib,io,json
from pathlib import Path
import sys,tempfile,unittest,zipfile
from unittest.mock import patch
import motor_diagnostic as d

class Arm:
    instance=None
    def __init__(self,*args):
        self.bus=self;self.closed=False;self.calls=[];Arm.instance=self
    def connect(self):pass
    def sync_read(self,name,**kwargs):
        assert kwargs=={'normalize':False,'num_retry':0}
        self.calls.append(name)
        return {n:0 for n in d.validate_calibration.__globals__['MOTORS']}
    def close(self):self.closed=True
    def report(self):return {'motor_register_write_commands_sent':0,'closed':self.closed}

class TestDiagnostic(unittest.TestCase):
    def run_case(self,fault=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data/robots').mkdir(parents=True)
            (root/'data/calibration/robots/so_follower').mkdir(parents=True)
            (root/'data/robots/lerobot.json').write_text('{"follower_port":"fake"}')
            (root/'data/calibration/robots/so_follower/lerobot.json').write_bytes((d.HERE/'reference_calibration.json').read_bytes())
            args=['diag','--lerobot-root',str(root),'--output',str(root/'out'),'--seconds','.01']
            class FailingArm(Arm):
                def sync_read(self,name,**kwargs):raise ConnectionError('injected read error')
            with patch.object(sys,'argv',args),patch.object(d,'ReadOnlyArm',FailingArm if fault else Arm),contextlib.redirect_stdout(io.StringIO()):
                code=d.main()
            report=json.loads((root/'out/summary.json').read_text())
            self.assertTrue(Arm.instance.closed)
            with zipfile.ZipFile(root/'out_return.zip') as z:self.assertIsNone(z.testzip())
            return code,report
    def test_normal_reads_and_cleanup(self):
        code,r=self.run_case();self.assertEqual(code,0);self.assertEqual(r['status'],'completed')
        self.assertEqual(set(Arm.instance.calls),set(d.STATIC+d.DYNAMIC))
    def test_read_errors_saved_and_closed(self):
        code,r=self.run_case(True);self.assertEqual(code,1);self.assertEqual(r['status'],'completed_with_read_errors')
        self.assertEqual(r['dynamic_read_errors'],len(d.DYNAMIC))

if __name__=='__main__':unittest.main(verbosity=2)
