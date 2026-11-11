"""Hardware-free regression checks; uses the installed Feetech SDK for packet tests."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from readonly_bus import MOTORS, ReadOnlyArm, ReadOnlyPacketGuard, validate_calibration


def packet(instruction,params=()):
    data=[255,255,1,len(params)+2,instruction,*params]
    return data+[(~sum(data[2:]))&255]

class GuardTests(unittest.TestCase):
    def test_all_instruction_opcodes(self):
        sent=[];guard=ReadOnlyPacketGuard(lambda p:sent.append(bytes(p)) or len(p))
        for instruction in range(256):
            if instruction in (1,2,130):guard(packet(instruction))
            else:
                with self.assertRaises(RuntimeError):guard(packet(instruction))
        self.assertEqual(len(sent),3);self.assertEqual(guard.blocked,253)
    def test_invalid_header_length_checksum(self):
        sent=[];guard=ReadOnlyPacketGuard(lambda p:sent.append(p))
        good=packet(2,[56,2])
        for bad in ([],good[:-1],good+[0],[0]+good[1:],good[:-1]+[good[-1]^1]):
            with self.assertRaises(RuntimeError):guard(bad)
        self.assertFalse(sent)
    def test_real_sdk_read_and_write(self):
        from scservo_sdk import PacketHandler
        class Port:
            is_using=False
            def __init__(self):self.sent=[];self.writePort=ReadOnlyPacketGuard(lambda p:self.sent.append(bytes(p)) or len(p))
            def clearPort(self):pass
            def setPacketTimeout(self,length):pass
        handler=PacketHandler(0);port=Port()
        handler.readTx(port,1,56,2);port.is_using=False
        handler.syncReadTx(port,56,2,[1,2,3,4,5,6],6);port.is_using=False
        self.assertEqual(len(port.sent),2)
        for call in (lambda:handler.write1ByteTxOnly(port,1,40,1),
                     lambda:handler.write2ByteTxOnly(port,1,42,2048),
                     lambda:handler.regWriteTxOnly(port,1,42,2,[0,8]),
                     lambda:handler.syncWriteTxOnly(port,42,2,[1,0,8],3),
                     lambda:handler.action(port,1)):
            port.is_using=False
            with self.assertRaises(RuntimeError):call()
        self.assertEqual(len(port.sent),2)

class ArmTests(unittest.TestCase):
    def test_connection_and_close_preserve_torque(self):
        calls=[]
        class Bus:
            is_calibrated=True
            def __init__(self):self.port_handler=SimpleNamespace(writePort=lambda p:len(p),is_open=False)
            def connect(self,handshake):calls.append(('connect',handshake));self.port_handler.is_open=True
            def sync_read(self,name,**kw):calls.append(('read',name));return {n:1 for n in MOTORS}
            def disconnect(self,disable_torque):calls.append(('disconnect',disable_torque));self.port_handler.is_open=False
        arm=ReadOnlyArm('unused',{},bus_factory=Bus)
        arm.connect();self.assertEqual(arm.read(),[1.0]*6);arm.close()
        self.assertEqual(calls[-1],('disconnect',False));self.assertTrue(arm.report()['torque_unchanged'])
        self.assertEqual(arm.guard.blocked,0)
    def test_calibration_mismatch_stops(self):
        reference=Path(__file__).with_name('reference_calibration.json')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'calibration.json';obj=json.loads(reference.read_text());obj['gripper']['homing_offset']+=1
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):validate_calibration(p,reference)

class InputTests(unittest.TestCase):
    def test_stale_state_rejected(self):
        import numpy as np
        from live_probe import Hardware
        hardware=Hardware(SimpleNamespace(),{})
        hardware.latest={'state':[0]*6,'timestamp':time.perf_counter()-1}
        cam=lambda:SimpleNamespace(thread=SimpleNamespace(is_alive=lambda:True),frame_lock=threading.Lock(),
                    latest_frame=np.zeros((480,640,3),dtype=np.uint8),latest_timestamp=time.perf_counter())
        hardware.cameras={'1':cam(),'3':cam()}
        with self.assertRaises(TimeoutError):hardware.snapshot()
    def test_stale_camera_rejected(self):
        import numpy as np
        from live_probe import Hardware
        hardware=Hardware(SimpleNamespace(),{})
        hardware.latest={'state':[0]*6,'timestamp':time.perf_counter()}
        cam=lambda:SimpleNamespace(thread=SimpleNamespace(is_alive=lambda:True),frame_lock=threading.Lock(),
                    latest_frame=np.zeros((480,640,3),dtype=np.uint8),latest_timestamp=time.perf_counter()-1)
        hardware.cameras={'1':cam(),'3':cam()}
        with self.assertRaises(TimeoutError):hardware.snapshot()

class PreviewTests(unittest.TestCase):
    def test_headless_http_preview_and_role_order(self):
        import urllib.request
        from unittest.mock import patch
        import numpy as np
        import cv2
        from live_probe import Preview
        with tempfile.TemporaryDirectory() as d, patch('webbrowser.open') as opened:
            preview=Preview(Path(d),True)
            try:
                left=np.zeros((480,640,3),dtype=np.uint8);left[:,:,2]=255
                right=np.zeros((480,640,3),dtype=np.uint8);right[:,:,0]=255
                preview.update({'images':{'3':left,'1':right}}, {'state':[0]*6}, 'first_preview.jpg')
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                page=opener.open(preview.url,timeout=5).read().decode()
                self.assertIn('EXTERNAL',page);self.assertTrue(opened.called)
                image=cv2.imread(str(Path(d)/'first_preview.jpg'))
                self.assertEqual(image.shape,(480,1280,3))
                self.assertGreater(int(image[200,200,0]),240)
                self.assertGreater(int(image[200,840,2]),240)
            finally:preview.close()

if __name__=='__main__':unittest.main(verbosity=2)
