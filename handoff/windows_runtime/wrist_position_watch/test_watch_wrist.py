import json
import unittest
from pathlib import Path
from readonly_bus import MOTORS,ReadOnlyPacketGuard
from watch_wrist import assess

CAL=json.loads((Path(__file__).parent/'reference_calibration.json').read_text(encoding='utf-8'))
class WatchTests(unittest.TestCase):
    def test_actual_and_target_readings(self):
        raw=dict(zip(MOTORS,[2023,777,3274,2751,1994,2046]));torque=dict.fromkeys(MOTORS,0)
        current=assess(raw,torque,CAL,.0301321335,.19780831)
        self.assertFalse(current['in_training_range']);self.assertAlmostEqual(current['training_zscore'],-23.485,places=2)
        raw['wrist_roll']=2046
        self.assertTrue(assess(raw,torque,CAL,.0301321335,.19780831)['in_training_range'])
    def test_any_torque_enabled_stops_and_bad_stats_rejected(self):
        raw=dict.fromkeys(MOTORS,2046);torque=dict.fromkeys(MOTORS,0);torque['wrist_roll']=1
        with self.assertRaisesRegex(RuntimeError,'Torque is not OFF'):assess(raw,torque,CAL,0,.2)
        torque['wrist_roll']=0
        with self.assertRaises(ValueError):assess(raw,torque,CAL,0,0)
    def test_serial_guard_blocks_all_writes(self):
        sent=[];guard=ReadOnlyPacketGuard(lambda p:sent.append(p))
        for packet in (bytes.fromhex('fffffe108328010101020103010401050106012a'),bytes.fromhex('ffff0104032800cf')):
            with self.assertRaises(RuntimeError):guard(packet)
        self.assertEqual(sent,[])

if __name__=='__main__':unittest.main()
