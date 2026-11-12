import unittest
from control_core import BoundedArm,MOTORS,sync_packet
from test_control import CAL,FakeBus
from test_fixed_wrist import SubsetBus
from set_d_trial import change

class Bus(FakeBus):
    def __init__(self,fail_d=False): super().__init__();self.frames=[];self.fail_d=fail_d
    def sync_write(self,name,values,**kw):
        ids=[MOTORS.index(n)+1 for n in values];vs=list(values.values())
        address={'Lock':55,'D_Coefficient':22}[name]
        if name=='D_Coefficient' and self.fail_d: raise OSError('injected D write failure')
        self.port_handler.writePort(sync_packet(address,1,vs,ids));self.frames.append((name,ids,vs))
        for n,v in values.items():self.data[name][MOTORS.index(n)]=v

class DTests(unittest.TestCase):
    def test_change_and_restore_only_active_ids(self):
        bus=Bus();arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);arm.connect()
        change(arm,0,{});self.assertEqual(bus.data['D_Coefficient'],[0,0,0,0,32,0]);self.assertEqual(bus.data['Lock'],[1]*6)
        change(arm,32,{});self.assertEqual(bus.data['D_Coefficient'],[32]*6)
        self.assertTrue(all(ids==[1,2,3,4,6] for _,ids,_ in bus.frames))
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_failure_relocks(self):
        bus=Bus(True);arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);arm.connect()
        with self.assertRaises(OSError):change(arm,0,{})
        self.assertEqual(bus.data['Lock'],[1]*6);self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_refuses_when_torque_on(self):
        bus=Bus();arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);arm.connect();bus.data['Torque_Enable'][2]=1
        with self.assertRaises(RuntimeError):change(arm,0,{})
        self.assertEqual(bus.frames,[])
    def test_d0_startup_keeps_wrist_passive(self):
        bus=SubsetBus();bus.data['D_Coefficient']=[0,0,0,0,32,0]
        arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);arm.connect()
        initial=arm.raw_read();arm.enable_aligned(initial)
        self.assertEqual(bus.data['Torque_Enable'],[1,1,1,1,0,1])
        self.assertTrue(all(5 not in ids for name,ids,_ in bus.written if name=='Goal_Position'))
        arm.release();self.assertEqual(bus.data['Torque_Enable'],[0]*6)

if __name__=='__main__':unittest.main()
