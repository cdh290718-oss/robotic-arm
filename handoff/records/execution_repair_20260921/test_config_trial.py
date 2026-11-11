import unittest
from config_trial import packet_for
from test_control import CAL,FakePort
from control_core import ExecutionGate

class ConfigGate(unittest.TestCase):
    def test_sdk_exact_packets(self):
        import scservo_sdk as sdk
        for name,value,address,length in [('P_Coefficient',20,21,1),('Homing_Offset',1722,31,2),('Min_Position_Limit',1064,9,2),('Lock',1,55,1)]:
            port=FakePort();gate=ExecutionGate(port.writePort,True);port.writePort=gate
            gate.expected=packet_for('elbow_flex',name,value,CAL)
            ph=sdk.PacketHandler(0)
            if length==1:ph.write1ByteTxOnly(port,3,address,value)
            else:ph.write2ByteTxOnly(port,3,address,value)
            self.assertIsNone(gate.expected)
            self.assertEqual(gate.write_packets,1)
    def test_no_other_gains_limits_torque_or_targets(self):
        for joint,name,value in [('elbow_flex','P_Coefficient',32),('shoulder_pan','P_Coefficient',20),('elbow_flex','Torque_Enable',1),('elbow_flex','Goal_Position',2000),('elbow_flex','Protection_Current',999),('elbow_flex','Homing_Offset',0)]:
            with self.assertRaises(ValueError):packet_for(joint,name,value,CAL)
    def test_negative_offset_encoding(self):
        packet=packet_for('shoulder_pan','Homing_Offset',-819,CAL)
        self.assertEqual(packet[6:8],bytes([0x33,0x0b]))
        from readonly_bus import ReadOnlyArm
        arm=ReadOnlyArm('fake',CAL)
        for joint,row in CAL.items():
            encoded=arm.bus._encode_sign('Homing_Offset',{row['id']:row['homing_offset']})[row['id']]
            self.assertEqual(packet_for(joint,'Homing_Offset',row['homing_offset'],CAL)[6:8],encoded.to_bytes(2,'little'))

if __name__=='__main__':unittest.main()
