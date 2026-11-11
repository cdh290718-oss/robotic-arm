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
WRIST_INDEX = 4
WRIST_DRIFT_TICKS = 11  # under 1 degree; monitored even for read-only predictions


def sync_packet(address, length, values, ids=None):
    params = [address, length]
    ids = list(range(1, len(values)+1)) if ids is None else list(ids)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError('Packet IDs and values must be distinct and aligned')
    for i, value in zip(ids, values):
        params += [i] + list(int(value).to_bytes(length, 'little'))
    data = [255, 255, 254, len(params) + 2, 0x83] + params
    return bytes(data + [(~sum(data[2:])) & 255])


class ExecutionGate:
    """At TX boundary: reads, or ONE explicitly scoped exact sync-write packet."""
    def __init__(self, write_port, writable=False, freeze_wrist=False):
        self.write_port = write_port
        self.writable = writable
        self.freeze_wrist = freeze_wrist
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
        if self.freeze_wrist and name == 'Torque_Enable' and any(values):
            raise ValueError('Fixed-wrist mode forbids explicit torque enable writes')
        ids = [1, 2, 3, 4, 6] if self.freeze_wrist and name == 'Goal_Position' else [1, 2, 3, 4, 5, 6]
        self.expected = sync_packet(address, length, [values[i-1] for i in ids], ids)
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
    def __init__(self, port, calibration, writable, bus_factory=None, freeze_wrist=False):
        super().__init__(port, calibration, bus_factory)
        original = self.guard._write_port
        self.guard = ExecutionGate(original, writable, freeze_wrist)
        self.bus.port_handler.writePort = self.guard
        self.may_be_enabled = False
        self.startup_complete = False
        self.events = []
        self.calibration = calibration
        self.freeze_wrist = freeze_wrist
        self.force_release = False

    def raw_read(self):
        values = self.bus.sync_read('Present_Position', normalize=False)
        raw = [int(values[n]) for n in MOTORS]
        if any(v < 0 or v > 4095 for v in raw):
            raise ValueError('Position outside single-turn 0..4095; refusing to execute')
        return raw

    def write_register(self, name, raw):
        if self.freeze_wrist and name == 'Goal_Position':
            self.assert_wrist_torque_off()
        with self.guard.permit(name, raw):
            # On this hardware even a goal-only write has preceded torque ON.
            # Arm cleanup BEFORE the serial call, including partial TX failures.
            if name == 'Goal_Position' or (name == 'Torque_Enable' and any(raw)):
                self.may_be_enabled = True
            values = {n:v for i,(n,v) in enumerate(zip(MOTORS,raw))
                      if not (self.freeze_wrist and name == 'Goal_Position' and i == WRIST_INDEX)}
            self.bus.sync_write(name, values, normalize=False, num_retry=0)
        if self.freeze_wrist and name == 'Goal_Position':
            self.assert_wrist_torque_off()

    def assert_wrist_torque_off(self):
        try:
            torque = self.bus.sync_read('Torque_Enable', normalize=False, num_retry=0)
            if int(torque['wrist_roll']) != 0:
                raise RuntimeError('Fixed wrist torque unexpectedly enabled')
        except BaseException:
            self.force_release = True
            raise

    def verify(self, name, expected):
        actual = self.bus.sync_read(name, normalize=False)
        names = [n for n in MOTORS if not (self.freeze_wrist and name == 'Goal_Position' and n == 'wrist_roll')]
        if any(int(actual[n]) != expected[MOTORS.index(n)] for n in names):
            raise RuntimeError(f'{name} readback mismatch: {actual}')

    def enable_aligned(self, raw, stop_requested=lambda: False):
        modes = self.bus.sync_read('Operating_Mode', normalize=False)
        if any(int(v) != 0 for v in modes.values()):
            raise ValueError('Expected position mode; no mode registers will be changed')
        self.verify('Torque_Enable', [0] * 6)
        self.verify('P_Coefficient', [16] * 6)
        self.verify('I_Coefficient', [0] * 6)
        self.verify('D_Coefficient', [32] * 6)
        self.verify('Lock', [1] * 6)
        grip_limit = self.bus.read('Max_Torque_Limit', 'gripper', normalize=False)
        if not 0 < grip_limit <= 500:
            raise RuntimeError('Gripper torque limit must be at most archived limit 500')
        fresh = self.raw_read()
        if max(abs(a-b) for a, b in zip(raw, fresh)) > 2:
            raise RuntimeError('Arm moved during preflight; position not stable')
        if stop_requested():
            raise RuntimeError('Stop requested before first goal write')
        self.write_register('Goal_Position', fresh)
        time.sleep(.02)
        self.verify('Goal_Position', fresh)
        expected_torque = [1,1,1,1,0,1] if self.freeze_wrist else [1] * 6
        self.verify('Torque_Enable', expected_torque)
        again = self.raw_read()
        if max(abs(a-b) for a, b in zip(again, fresh)) > 2:
            raise RuntimeError('Arm moved during goal-triggered activation')
        if stop_requested():
            raise RuntimeError('Stop requested before torque enable')
        time.sleep(.02)
        self.verify('Torque_Enable', expected_torque)
        if max(abs(a-b) for a,b in zip(self.raw_read(),fresh)) > 2:
            raise RuntimeError('Arm moved after activation')
        self.startup_complete = True
        self.events.append({'event': 'goal_triggered_activation_verified', 'time': time.perf_counter(), 'raw': fresh})
        return fresh

    def hold(self):
        raw = self.raw_read()
        self.write_register('Goal_Position', raw)
        self.verify('Goal_Position', raw)
        torques = self.bus.sync_read('Torque_Enable', normalize=False)
        expected = [1,1,1,1,0,1] if self.freeze_wrist else [1]*6
        result = {'goal_readback_confirmed': True, 'raw_goal': raw, 'torque': torques,
                  'all_torque_enabled': all(int(v) == 1 for v in torques.values()),
                  'all_expected_torque_states': [int(torques[n]) for n in MOTORS] == expected,
                  'fixed_wrist_torque_off': int(torques['wrist_roll']) == 0 if self.freeze_wrist else None,
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
        if not self.startup_complete or self.force_release:
            # Startup began torque OFF; restore that state, without another goal.
            reason = 'release_fixed_wrist_fault' if self.force_release else 'release_failed_startup'
            self.release()
            return {'action': reason, 'confirmed': True}
        result = self.hold()
        return {'action': 'hold', 'confirmed': result['all_expected_torque_states'], **result}

    def report(self):
        return {'torque_before': self.torque_before, 'torque_after': self.torque_after,
                'write_attempts': self.guard.write_attempts, 'write_packets': self.guard.write_packets,
                'packet_counts': dict(self.guard.counts), 'blocked_packets': self.guard.blocked,
                'events': self.events, 'normal_robot_connect_called': False,
                'fixed_wrist': self.freeze_wrist}


class Limiter:
    """Limit raw setpoints; one raw tick maximum per 30 Hz step (~2.64 deg/s).

    Flooring preserves the 5 deg/s upper bound without quantization overshoot.
    This limits commanded setpoints, not independently measured motor velocity.
    """
    def __init__(self, calibration, initial, freeze_wrist=False):
        self.cal = calibration
        self.initial = list(initial)
        self.last = list(initial)
        self.freeze_wrist = freeze_wrist
        excursion = math.floor(EXCURSION_DEG / DEG_PER_TICK)
        self.lo = [max(calibration[n]['range_min'], v-excursion) for n, v in zip(MOTORS, initial)]
        self.hi = [min(calibration[n]['range_max'], v+excursion) for n, v in zip(MOTORS, initial)]
        if self.freeze_wrist:
            self.lo[WRIST_INDEX] = self.hi[WRIST_INDEX] = initial[WRIST_INDEX]
        if any(not lo <= v <= hi for lo, v, hi in zip(self.lo, initial, self.hi)):
            details = '; '.join(f'{n}: raw={v}, allowed={lo}..{hi}'
                                for n,v,lo,hi in zip(MOTORS,initial,self.lo,self.hi) if not lo<=v<=hi)
            raise ValueError('Initial joint position outside saved calibration range: '+details)
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
        if any(not self.lo[i]-2 <= raw[i] <= self.hi[i]+2 for i in range(5) if not (self.freeze_wrist and i==WRIST_INDEX)):
            raise RuntimeError('Measured position outside trial envelope')
        if self.freeze_wrist and abs(raw[WRIST_INDEX]-self.initial[WRIST_INDEX])>WRIST_DRIFT_TICKS:
            raise RuntimeError('Fixed wrist moved more than 11 encoder ticks')
        if enforce_tracking and max(abs(raw[i]-self.last[i])*DEG_PER_TICK for i in range(5) if not (self.freeze_wrist and i==WRIST_INDEX)) > TRACKING_ERROR_DEG:
            raise RuntimeError('Measured tracking error exceeds 4 degrees')
        if abs(raw[5]-self.initial[5]) > 15:
            raise RuntimeError('Gripper moved despite fixed target')


class GraspLimiter(Limiter):
    """Full saved workspace and moving gripper; retain body tracking/speed guards."""
    def __init__(self, calibration, initial, freeze_wrist=False):
        self.cal=calibration
        self.initial=list(initial)
        self.last=list(initial)
        self.freeze_wrist=freeze_wrist
        self.lo=[calibration[n]['range_min'] for n in MOTORS]
        self.hi=[calibration[n]['range_max'] for n in MOTORS]
        if self.freeze_wrist:
            self.lo[WRIST_INDEX]=self.hi[WRIST_INDEX]=initial[WRIST_INDEX]
        if any(not lo<=v<=hi for lo,v,hi in zip(self.lo,initial,self.hi)):
            details = '; '.join(f'{n}: raw={v}, allowed={lo}..{hi}'
                                for n,v,lo,hi in zip(MOTORS,initial,self.lo,self.hi) if not lo<=v<=hi)
            raise ValueError('Initial position outside saved calibration range: '+details)

    def command(self, action, dt):
        previous_grip=self.last[5]
        raw,clipped=super().command(action,dt)
        c=self.cal['gripper']
        desired=int(float(action[5])/100*(c['range_max']-c['range_min'])+c['range_min'])
        bounded=max(self.lo[5],min(self.hi[5],desired))
        # <=5 normalized percentage points/s, rounded down; never jump to target.
        step=math.floor(5*min(dt,1/HZ)*(c['range_max']-c['range_min'])/100)
        raw[5]=max(previous_grip-step,min(previous_grip+step,bounded))
        clipped[5]=raw[5]!=desired
        self.last=raw
        return raw,clipped

    def check_measured(self,raw,enforce_tracking=True):
        if any(not self.lo[i]-2<=raw[i]<=self.hi[i]+2 for i in range(6) if not (self.freeze_wrist and i==WRIST_INDEX)):
            raise RuntimeError('Measured position outside saved grasp workspace')
        if self.freeze_wrist and abs(raw[WRIST_INDEX]-self.initial[WRIST_INDEX])>WRIST_DRIFT_TICKS:
            raise RuntimeError('Fixed wrist moved more than 11 encoder ticks')
        if enforce_tracking and max(abs(raw[i]-self.last[i])*DEG_PER_TICK for i in range(5) if not (self.freeze_wrist and i==WRIST_INDEX))>TRACKING_ERROR_DEG:
            raise RuntimeError('Measured tracking error exceeds 4 degrees')
        if enforce_tracking and abs(raw[5]-self.last[5])*100/(self.hi[5]-self.lo[5])>5:
            raise RuntimeError('Gripper tracking error exceeds 5 percentage points')


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
