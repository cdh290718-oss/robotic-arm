import unittest
from control_core import BoundedArm, MOTORS
from test_control import CAL
from test_fixed_wrist import SubsetBus
from threshold import run

class Clock:
    def __init__(self): self.t=0
    def now(self): return self.t
    def sleep(self, v): self.t+=v

class SimBus(SubsetBus):
    def __init__(self, mode='move'):
        super().__init__(mode=='wrist')
        self.mode=mode
        self.data['Present_Position'][2]=(CAL['elbow_flex']['range_min']+CAL['elbow_flex']['range_max'])//2
        for n in ('Status','Present_Current','Present_Load','Present_Voltage'): self.data[n]=[0]*6
    def sync_write(self,name,values,**kw):
        if name=='Goal_Position' and self.mode=='write_failure' and self.written:
            raise ConnectionError('simulated partial transmission')
        super().sync_write(name,values,**kw)
        if name=='Goal_Position' and self.mode=='move':
            pos=self.data['Present_Position'][2]
            delta=values['elbow_flex']-pos
            if abs(delta)>=12:
                self.data['Present_Position'][2]+=3 if delta>0 else -3

class Tests(unittest.TestCase):
    def simulate(self, mode):
        bus=SimBus(mode); arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True); c=Clock()
        report={'status':'starting','trials':[]}
        run(arm,CAL,report,clock=c.now,sleep=c.sleep,stopped=lambda:False)
        return bus,report
    def test_both_directions_and_no_configuration_writes(self):
        bus,r=self.simulate('move')
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual([t['status'] for t in r['trials']],['started','started'])
        self.assertEqual([t['start_error_ticks'] for t in r['trials']],[9,-9])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertTrue(all(5 not in ids for name,ids,_ in bus.written if name=='Goal_Position'))
        self.assertTrue(all(name in ('Goal_Position','Torque_Enable') for name,_,_ in bus.written))
    def test_no_movement_censors_both_directions(self):
        bus,r=self.simulate('stuck')
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual([t['status'] for t in r['trials']],['not_observed_within_40_ticks']*2)
        self.assertTrue(all(abs(t['samples'][-2]['elbow_error_ticks'])==40 for t in r['trials']))
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_unexpected_wrist_activation_releases(self):
        bus,r=self.simulate('wrist')
        self.assertEqual(r['status'],'failed')
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertTrue(r.get('release_confirmed'))
    def test_write_failure_releases_and_does_not_reverse(self):
        bus,r=self.simulate('write_failure')
        self.assertEqual(r['status'],'failed')
        self.assertEqual(len(r['trials']),1)
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertTrue(r.get('release_confirmed'))

if __name__=='__main__': unittest.main()
