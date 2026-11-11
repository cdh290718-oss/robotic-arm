import json
import math
from pathlib import Path
import unittest
import numpy as np
from control_core import (ExecutionGate, BoundedArm, Limiter, MOTORS, DEG_PER_TICK, action_at, sync_packet)
from bounded_run import SharedObservation

CAL = json.loads((Path(__file__).parent/'reference_calibration.json').read_text())
INITIAL = [1964,778,3274,2908,2042,2046]


class FakePort:
    is_open = True
    is_using = False
    def __init__(self): self.sent=[]
    def writePort(self, packet): self.sent.append(bytes(packet)); return len(packet)
    def clearPort(self): pass
    def setPacketTimeout(self, *args): pass
    def closePort(self): self.is_open=False


class FakeBus:
    is_calibrated = True
    def __init__(self):
        self.port_handler=FakePort()
        self.data={'Present_Position':list(INITIAL),'Goal_Position':[123]*6,
                   'Torque_Enable':[0]*6, 'Operating_Mode':[0]*6}
        self.events=[]
        self.bad_readback=False
    def connect(self, **kwargs): self.events.append('connect')
    def sync_read(self, name, **kwargs):
        self.events.append('read_'+name)
        values = self.data[name].copy()
        if name=='Goal_Position' and self.bad_readback: values[0]+=1
        return dict(zip(MOTORS,values))
    def sync_write(self,name,values,**kwargs):
        raw=[values[n] for n in MOTORS]
        address,length=(42,2) if name=='Goal_Position' else (40,1)
        self.port_handler.writePort(sync_packet(address,length,raw))
        self.events.append('write_'+name)
        self.data[name]=raw
    def disconnect(self,disable_torque=True):
        assert disable_torque is False
        self.events.append('disconnect_without_torque_change')


class TestGate(unittest.TestCase):
    def test_unauthorized_all_opcodes(self):
        port=FakePort(); gate=ExecutionGate(port.writePort,True)
        for opcode in range(256):
            p=[255,255,1,2,opcode]; p.append((~sum(p[2:]))&255)
            if opcode in (1,2,0x82): gate(p)
            else:
                with self.assertRaises(RuntimeError): gate(p)
        self.assertEqual(len(port.sent),3)

    def test_exact_single_use_and_readonly(self):
        port=FakePort(); gate=ExecutionGate(port.writePort,True)
        packet=sync_packet(42,2,INITIAL)
        with gate.permit('Goal_Position',INITIAL): gate(packet)
        with self.assertRaises(RuntimeError): gate(packet)
        with self.assertRaises(RuntimeError):
            with gate.permit('Goal_Position',INITIAL): gate(sync_packet(42,2,[2048]*6))
        with self.assertRaises(ValueError):
            with gate.permit('Homing_Offset',[0]*6): pass
        with self.assertRaises(ValueError):
            with gate.permit('Torque_Enable',[2]*6): pass
        gate.writable=False
        with self.assertRaises(RuntimeError):
            with gate.permit('Torque_Enable',[1]*6): pass
        self.assertEqual(gate.write_packets,1)

    def test_real_sdk_packets(self):
        import scservo_sdk as sdk
        port=FakePort(); underlying=port.writePort
        gate=ExecutionGate(underlying,True); port.writePort=gate
        ph=sdk.PacketHandler(0)
        params=[]
        for i,v in enumerate(INITIAL,1): params.extend([i,v&255,v>>8])
        with gate.permit('Goal_Position',INITIAL):
            ph.syncWriteTxOnly(port,42,2,params,len(params))
        port.is_using=False
        torque=[]
        for i in range(1,7): torque.extend([i,1])
        with gate.permit('Torque_Enable',[1]*6):
            ph.syncWriteTxOnly(port,40,1,torque,len(torque))
        for call in (lambda:ph.write1ByteTxOnly(port,1,40,1),
                     lambda:ph.write2ByteTxOnly(port,1,31,123),
                     lambda:ph.syncWriteTxOnly(port,42,2,params,len(params)),
                     lambda:ph.action(port,1)):
            port.is_using=False
            with self.assertRaises(RuntimeError): call()
        self.assertEqual(gate.write_packets,2)


class TestArm(unittest.TestCase):
    def make(self,writable=True):
        bus=FakeBus(); arm=BoundedArm('fake',CAL,writable,bus_factory=lambda:bus)
        arm.connect(); return arm,bus
    def test_alignment_order_and_hold(self):
        arm,bus=self.make()
        arm.enable_aligned(INITIAL)
        self.assertLess(bus.events.index('write_Goal_Position'),bus.events.index('write_Torque_Enable'))
        self.assertLess(bus.events.index('read_Goal_Position'),bus.events.index('write_Torque_Enable'))
        self.assertTrue(arm.hold()['all_torque_enabled'])
        arm.close(); self.assertEqual(bus.data['Torque_Enable'],[1]*6)
    def test_bad_alignment_never_enables(self):
        arm,bus=self.make(); bus.bad_readback=True
        with self.assertRaises(RuntimeError): arm.enable_aligned(INITIAL)
        self.assertNotIn('write_Torque_Enable',bus.events)
    def test_stop_before_enable(self):
        arm,bus=self.make()
        with self.assertRaises(RuntimeError): arm.enable_aligned(INITIAL,lambda:True)
        self.assertNotIn('write_Torque_Enable',bus.events)
    def test_readonly_has_no_writes(self):
        arm,bus=self.make(False)
        arm.raw_read()
        with self.assertRaises(RuntimeError): arm.write_register('Goal_Position',INITIAL)
        arm.close(); self.assertEqual(arm.guard.write_attempts,0)
    def test_wrong_mode_and_existing_torque_never_enable(self):
        for register,value in [('Operating_Mode',1),('Torque_Enable',1)]:
            arm,bus=self.make();bus.data[register]=[value]*6
            with self.assertRaises((RuntimeError,ValueError)): arm.enable_aligned(INITIAL)
            self.assertFalse(any(e.startswith('write_') for e in bus.events))

    def test_release_only_torque(self):
        arm,bus=self.make(); bus.data['Torque_Enable']=[1]*6
        arm.release()
        self.assertEqual([e for e in bus.events if e.startswith('write_')],['write_Torque_Enable'])


class TestLimits(unittest.TestCase):
    def test_limits_quantization_and_fixed_gripper(self):
        lim=Limiter(CAL,INITIAL)
        for t in range(500):
            previous=lim.last.copy()
            raw,clipped=lim.command([10000*(-1 if t%70<35 else 1)]*6,1/30)
            self.assertTrue(all(abs(a-b)<=1 for a,b in zip(raw,previous)))
            self.assertTrue(all(lo<=v<=hi for lo,v,hi in zip(lim.lo,raw,lim.hi)))
            self.assertTrue(all(abs(v-i)*DEG_PER_TICK<=8 for v,i in zip(raw[:5],INITIAL[:5])))
            self.assertEqual(raw[5],INITIAL[5])
        self.assertLessEqual(lim.hi[4]-lim.lo[4],9)
    def test_bad_actions_stall_tracking_and_calibration(self):
        lim=Limiter(CAL,INITIAL)
        for action in ([math.nan]*6,[math.inf]*6,[0]*5):
            with self.assertRaises(ValueError): lim.command(action,.033)
        with self.assertRaises(TimeoutError): lim.command([0]*6,.151)
        bad=INITIAL.copy();bad[0]+=60
        with self.assertRaises(RuntimeError): lim.check_measured(bad)
        bad=INITIAL.copy();bad[4]=1000
        with self.assertRaises(ValueError): Limiter(CAL,bad)
    def test_age_skips_old_actions_and_expires(self):
        actions=np.arange(300).reshape(50,6)
        chunk={'observation_time':100,'actions':actions}
        action,index=action_at(chunk,101.0)
        self.assertEqual(index,30);np.testing.assert_equal(action,actions[30])
        self.assertIsNone(action_at(chunk,102)[0])
        with self.assertRaises(ValueError): action_at(chunk,99)
        with self.assertRaises(ValueError): action_at({'observation_time':100,'actions':[[math.nan]*6]*50},101)
    def test_units_match_lerobot(self):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if i==6 else MotorNormMode.DEGREES) for i,n in enumerate(MOTORS,1)}
        # Developer SDK is unpacked in /tmp, without installed distribution metadata.
        # Bypass only its metadata check; use actual SDK and normalization implementation.
        from unittest.mock import patch
        import scservo_sdk
        with patch('lerobot.motors.feetech.feetech.require_package'), patch('lerobot.motors.feetech.feetech.scs',scservo_sdk):
            bus=FeetechMotorsBus(port='fake',motors=motors,calibration={n:MotorCalibration(**c) for n,c in CAL.items()})
        normalized=bus._normalize(dict(enumerate(INITIAL,1)))
        np.testing.assert_allclose(Limiter(CAL,INITIAL).state(INITIAL),list(normalized.values()),atol=1e-10)
    def test_shared_memory_latest_and_nonblocking(self):
        import multiprocessing as mp
        shared=SharedObservation(mp.get_context('spawn'))
        row={'images':{k:np.full((480,640,3),i,np.uint8) for i,k in enumerate(('3','1'))},
             'state':[1,2,3,4,5,6],'state_timestamp':10,'image_timestamps':{'1':9,'3':8}}
        self.assertTrue(shared.put(row,1))
        got=shared.get(0); self.assertEqual(got['sequence'],1)
        np.testing.assert_equal(got['state'],row['state'])
        self.assertEqual(got['image_timestamps'],row['image_timestamps'])
        self.assertIsNone(shared.get(1))
        shared.lock.acquire()
        try: self.assertFalse(shared.put(row,2))
        finally: shared.lock.release()

if __name__=='__main__': unittest.main(verbosity=2)
