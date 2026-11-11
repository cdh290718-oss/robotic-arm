"""SO101 read-only adapter. No SOFollower.connect/configure/calibrate calls."""
from collections import Counter
import json
from pathlib import Path

MOTORS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

class ReadOnlyPacketGuard:
    """Allow only Feetech ping, read, sync-read packets at the serial TX boundary."""
    def __init__(self, write_port):
        self._write_port = write_port
        self.counts = Counter()
        self.blocked = 0

    def __call__(self, packet):
        data = bytes(packet)
        valid = (len(data) >= 6 and data[:2] == b'\xff\xff' and len(data) == data[3] + 4
                 and (sum(data[2:]) & 255) == 255)
        if not valid or data[4] not in (0x01, 0x02, 0x82):
            self.blocked += 1
            raise RuntimeError('Blocked non-read or malformed motor packet. No bytes transmitted.')
        self.counts[f'0x{data[4]:02x}'] += 1
        return self._write_port(packet)


def validate_calibration(local_path, reference_path):
    local = json.loads(Path(local_path).read_text(encoding='utf-8-sig'))
    reference = json.loads(Path(reference_path).read_text(encoding='utf-8-sig'))
    if local != reference:
        raise ValueError(f'Calibration differs from the training archive: {local_path}. No calibration will be written.')
    if set(local) != set(MOTORS):
        raise ValueError('Calibration motor names differ from the six trained joints')
    for i,name in enumerate(MOTORS,1):
        row=local[name]
        if row['id'] != i or row['range_min'] >= row['range_max']:
            raise ValueError(f'Invalid calibration: {name}')
    return local


class ReadOnlyArm:
    def __init__(self, port, calibration, bus_factory=None):
        if bus_factory is None:
            from lerobot.motors import Motor, MotorCalibration, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
            motors={n: Motor(i, 'sts3215', MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES)
                    for i,n in enumerate(MOTORS,1)}
            bus_factory=lambda: FeetechMotorsBus(port=port, motors=motors,
                                calibration={n:MotorCalibration(**v) for n,v in calibration.items()})
        self.bus=bus_factory()
        self.guard=ReadOnlyPacketGuard(self.bus.port_handler.writePort)
        self.bus.port_handler.writePort=self.guard
        self.torque_before=None
        self.torque_after=None
        self.connected=False

    def connect(self):
        # Serial handshake pings/reads only. Guard is installed BEFORE opening the port.
        self.bus.connect(handshake=True)
        self.connected=True
        if not self.bus.is_calibrated:
            raise ValueError('On-device calibration differs from the saved training calibration. Stopping without writes.')
        self.torque_before=self.bus.sync_read('Torque_Enable', normalize=False)

    def read(self):
        # Identical normalization to SOFollower(use_degrees=True): five angles + gripper [0,100].
        values=self.bus.sync_read('Present_Position')
        return [float(values[name]) for name in MOTORS]

    def close(self):
        if self.connected:
            try:
                self.torque_after=self.bus.sync_read('Torque_Enable', normalize=False)
            finally:
                self.bus.disconnect(disable_torque=False)
                self.connected=False
        elif self.bus.port_handler.is_open:
            self.bus.port_handler.closePort()

    def report(self):
        return {'allowed_packet_counts':dict(self.guard.counts),'blocked_packets':self.guard.blocked,
                'torque_before':self.torque_before,'torque_after':self.torque_after,
                'torque_unchanged':self.torque_before==self.torque_after if self.torque_before is not None and self.torque_after is not None else None,
                'units':'five body joints in calibrated degrees; gripper normalized 0..100',
                'normal_robot_connect_called':False,'motor_register_write_commands_sent':0}
