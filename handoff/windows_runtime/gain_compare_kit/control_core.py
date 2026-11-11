"""Bounded SO101 position commands; no hardware imports at module import time."""
from collections import Counter
from contextlib import contextmanager
import math
import time
from readonly_bus import MOTORS, ReadOnlyArm

HZ = 30
DEG_PER_TICK = 360 / 4095
SPEED_DEG_S = 5.0
EXCURSION_DEG = 8.0
MAX_LOOP_GAP = .150
MAX_INPUT_AGE = .250
MAX_RESULT_AGE = 1.40
NO_PLAN_TIMEOUT = 2.0
TRACKING_ERROR_DEG = 4.0


def sync_packet(address, length, values):
    params = [address, length]
    for i, value in enumerate(values, 1):
        params += [i] + list(int(value).to_bytes(length, 'little'))
    data = [255, 255, 254, len(params) + 2, 0x83] + params
    return bytes(data + [(~sum(data[2:])) & 255])


class ExecutionGate:
    """At TX boundary: reads, or ONE explicitly scoped exact sync-write packet."""
    def __init__(self, write_port, writable=False):
        self.write_port = write_port
        self.writable = writable
        self.expected = None
        self.counts = Counter()
        self.blocked = 0
        self.write_attempts = 0
        self.write_packets = 0

    @contextmanager
    def permit(self, name, values):
        if not self.writable or self.expected is not None:
            raise RuntimeError('Write gate is closed or already in a transaction')
        if len(values) != 6 or any(type(v) is not int for v in values):
            raise ValueError('Six integer motor values required')
        if name == 'Goal_Position' and all(0 <= v <= 4095 for v in values):
            address, length = 42, 2
        elif name == 'Torque_Enable' and all(v in (0, 1) for v in values):
            address, length = 40, 1
        else:
            raise ValueError('Register or value not allowed')
        self.expected = sync_packet(address, length, values)
        try:
            yield
            if self.expected is not None:
                raise RuntimeError('Authorized write did not reach the serial boundary')
        finally:
            self.expected = None

    def __call__(self, packet):
        data = bytes(packet)
        valid = (len(data) >= 6 and data[:2] == b'\xff\xff' and
                 len(data) == data[3] + 4 and sum(data[2:]) & 255 == 255)
        if not valid:
            self.blocked += 1
            raise RuntimeError('Malformed motor packet blocked')
        if data[4] not in (1, 2, 0x82):
            if not self.writable or self.expected is None or data != self.expected:
                self.blocked += 1
                raise RuntimeError('Unapproved motor write blocked before transmission')
            self.expected = None  # consume even if the underlying TX fails
            self.write_attempts += 1
            count = self.write_port(packet)
            if count != len(data):
                raise IOError('Incomplete serial write; hardware state uncertain')
            self.write_packets += 1
        else:
            count = self.write_port(packet)
        self.counts[f'0x{data[4]:02x}'] += 1
        return count


class BoundedArm(ReadOnlyArm):
    def __init__(self, port, calibration, writable, bus_factory=None):
        super().__init__(port, calibration, bus_factory)
        original = self.guard._write_port
        self.guard = ExecutionGate(original, writable)
        self.bus.port_handler.writePort = self.guard
        self.may_be_enabled = False
        self.startup_complete = False
        self.events = []
        self.calibration = calibration

    def raw_read(self):
        values = self.bus.sync_read('Present_Position', normalize=False)
        raw = [int(values[n]) for n in MOTORS]
        if any(v < 0 or v > 4095 for v in raw):
            raise ValueError('Position outside single-turn 0..4095; refusing to execute')
        return raw

    def write_register(self, name, raw):
        with self.guard.permit(name, raw):
            # On this hardware even a goal-only write has preceded torque ON.
            # Arm cleanup BEFORE the serial call, including partial TX failures.
            if name == 'Goal_Position' or (name == 'Torque_Enable' and any(raw)):
                self.may_be_enabled = True
            self.bus.sync_write(name, dict(zip(MOTORS, raw)), normalize=False, num_retry=0)

    def verify(self, name, expected):
        actual = self.bus.sync_read(name, normalize=False)
        if [int(actual[n]) for n in MOTORS] != expected:
            raise RuntimeError(f'{name} readback mismatch: {actual}')

    def enable_aligned(self, raw, stop_requested=lambda: False):
        modes = self.bus.sync_read('Operating_Mode', normalize=False)
        if any(int(v) != 0 for v in modes.values()):
            raise ValueError('Expected position mode; no mode registers will be changed')
        self.verify('Torque_Enable', [0] * 6)
        fresh = self.raw_read()
        if max(abs(a-b) for a, b in zip(raw, fresh)) > 2:
            raise RuntimeError('Arm moved during preflight; position not stable')
        self.write_register('Goal_Position', fresh)
        self.verify('Goal_Position', fresh)
        self.verify('Torque_Enable', [0] * 6)
        again = self.raw_read()
        if max(abs(a-b) for a, b in zip(again, fresh)) > 2:
            raise RuntimeError('Arm moved before torque enable')
        if stop_requested():
            raise RuntimeError('Stop requested before torque enable')
        self.may_be_enabled = True  # partial broadcast failure also needs cleanup
        self.write_register('Torque_Enable', [1] * 6)
        self.verify('Torque_Enable', [1] * 6)
        self.startup_complete = True
        self.events.append({'event': 'aligned_then_enabled', 'time': time.perf_counter(), 'raw': fresh})
        return fresh

    def hold(self):
        raw = self.raw_read()
        self.write_register('Goal_Position', raw)
        self.verify('Goal_Position', raw)
        torques = self.bus.sync_read('Torque_Enable', normalize=False)
        result = {'goal_readback_confirmed': True, 'raw_goal': raw, 'torque': torques,
                  'all_torque_enabled': all(int(v) == 1 for v in torques.values()),
                  'note': 'Position hold commanded; not a hardware emergency stop or physical stability proof.'}
        self.events.append({'event': 'stop_hold', **result})
        return result

    def release(self):
        self.write_register('Torque_Enable', [0] * 6)
        self.verify('Torque_Enable', [0] * 6)
        self.may_be_enabled = False
        self.startup_complete = False
        self.events.append({'event': 'torque_released'})

    def cleanup_motion(self):
        if not self.may_be_enabled:
            return {'action': 'none', 'confirmed': True}
        if not self.startup_complete:
            # Startup began torque OFF; restore that state, without another goal.
            self.release()
            return {'action': 'release_failed_startup', 'confirmed': True}
        result = self.hold()
        return {'action': 'hold', 'confirmed': result['all_torque_enabled'], **result}

    def report(self):
        return {'torque_before': self.torque_before, 'torque_after': self.torque_after,
                'write_attempts': self.guard.write_attempts, 'write_packets': self.guard.write_packets,
                'packet_counts': dict(self.guard.counts), 'blocked_packets': self.guard.blocked,
                'events': self.events, 'normal_robot_connect_called': False}


class Limiter:
    """Limit raw setpoints; one raw tick maximum per 30 Hz step (~2.64 deg/s).

    Flooring preserves the 5 deg/s upper bound without quantization overshoot.
    This limits commanded setpoints, not independently measured motor velocity.
    """
    def __init__(self, calibration, initial):
        self.cal = calibration
        self.initial = list(initial)
        self.last = list(initial)
        excursion = math.floor(EXCURSION_DEG / DEG_PER_TICK)
        self.lo = [max(calibration[n]['range_min'], v-excursion) for n, v in zip(MOTORS, initial)]
        self.hi = [min(calibration[n]['range_max'], v+excursion) for n, v in zip(MOTORS, initial)]
        if any(not lo <= v <= hi for lo, v, hi in zip(self.lo, initial, self.hi)):
            raise ValueError('Initial joint position outside saved calibration range')
        self.lo[5] = self.hi[5] = initial[5]  # first trial never opens/closes gripper

    def state(self, raw):
        result = []
        for i, n in enumerate(MOTORS):
            c = self.cal[n]
            if i == 5:
                result.append((max(c['range_min'], min(c['range_max'], raw[i]))-c['range_min']) * 100 /
                              (c['range_max']-c['range_min']))
            else:
                result.append((raw[i]-(c['range_min']+c['range_max'])/2)*DEG_PER_TICK)
        return result

    def command(self, action, dt):
        if len(action) != 6 or not all(math.isfinite(float(v)) for v in action):
            raise ValueError('Invalid/nonfinite predicted action')
        if not 0 < dt <= MAX_LOOP_GAP:
            raise TimeoutError('Control interval exceeded 150 ms')
        step = math.floor(SPEED_DEG_S * min(dt, 1/HZ) / DEG_PER_TICK)
        raw = []
        limited = []
        for i, n in enumerate(MOTORS):
            c = self.cal[n]
            desired = (int(float(action[i])/100*(c['range_max']-c['range_min'])+c['range_min']) if i == 5
                       else int(float(action[i])/DEG_PER_TICK+(c['range_min']+c['range_max'])/2))
            bounded = max(self.lo[i], min(self.hi[i], desired))
            value = max(self.last[i]-step, min(self.last[i]+step, bounded))
            raw.append(value)
            limited.append(value != desired)
        self.last = raw
        return raw, limited

    def check_measured(self, raw, enforce_tracking=True):
        if any(not self.lo[i]-2 <= raw[i] <= self.hi[i]+2 for i in range(5)):
            raise RuntimeError('Measured position outside trial envelope')
        if enforce_tracking and max(abs(raw[i]-self.last[i])*DEG_PER_TICK for i in range(5)) > TRACKING_ERROR_DEG:
            raise RuntimeError('Measured tracking error exceeds 4 degrees')
        if abs(raw[5]-self.initial[5]) > 15:
            raise RuntimeError('Gripper moved despite fixed target')


def action_at(chunk, now):
    import numpy as np
    actions = np.asarray(chunk['actions'])
    if actions.shape != (50, 6) or not np.isfinite(actions).all():
        raise ValueError('Malformed action chunk')
    age = now - float(chunk['observation_time'])
    if age < 0:
        raise ValueError('Future observation timestamp')
    index = int(age * HZ)
    if index >= 50:
        return None, index
    return actions[index], index
