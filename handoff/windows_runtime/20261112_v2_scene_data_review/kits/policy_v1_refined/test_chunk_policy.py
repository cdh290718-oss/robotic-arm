import unittest
from bounded_run import choose_plan

def chunk(identifier, observation):
    return {'id':identifier,'observation_time':observation,
            'actions':[[float(identifier)]*6 for _ in range(50)]}

class ChunkPolicyTests(unittest.TestCase):
    def run_select(self,a,p,t,now=100.5,n=25):
        rejected=[]
        out=choose_plan(a,p,t,now,n,30,1.4,rejected)
        return out,rejected
    def test_wait_then_promote_at_25(self):
        a,p=chunk(1,100.0),chunk(2,100.25)
        out,_=self.run_select(a,p,24)
        self.assertEqual(out[0]['id'],1);self.assertEqual(out[1]['id'],2)
        out,_=self.run_select(a,p,25)
        self.assertEqual(out[0]['id'],2);self.assertIsNone(out[1]);self.assertEqual(out[2],0)
        self.assertEqual(out[5]['reason'],'minimum_ticks');self.assertEqual(out[4],7)
    def test_pending_age_rechecked_at_promotion(self):
        a,p=chunk(1,100.0),chunk(2,99.0)
        out,rejected=self.run_select(a,p,25)
        self.assertEqual(out[0]['id'],1);self.assertIsNone(out[1])
        self.assertEqual(rejected[0]['reason'],'expired_while_pending')
    def test_expired_active_promotes_fresh_pending_before_25(self):
        a,p=chunk(1,98.0),chunk(2,100.0)
        out,_=self.run_select(a,p,5)
        self.assertEqual(out[0]['id'],2);self.assertEqual(out[5]['reason'],'active_expired')
    def test_no_usable_plan_holds(self):
        out,_=self.run_select(chunk(1,98.0),None,5)
        self.assertIsNone(out[3]);self.assertGreaterEqual(out[4],50)
    def test_default_immediate_replacement(self):
        out,_=self.run_select(chunk(1,100.0),chunk(2,100.2),1,n=0)
        self.assertEqual(out[0]['id'],2)

if __name__=='__main__':unittest.main()
