"""Bounded elbow breakaway observation. Importing this module never accesses hardware."""
import argparse
import json
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

from control_core import BoundedArm, DEG_PER_TICK, GraspLimiter, MOTORS
from readonly_bus import validate_calibration

HERE = Path(__file__).resolve().parent
CAL_PATH = Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json')
OFF = [0] * 6
EXPECTED = [1, 1, 1, 1, 0, 1]


def stop_requested():
    try:
        import msvcrt
        return msvcrt.kbhit() and msvcrt.getwch() in ('\x1b', ' ')
    except ImportError:
        return False


def sample(arm, limiter, initial, goal, trial, phase, clock):
    raw = arm.raw_read()
    row = {'time': clock(), 'phase': phase, 'target': goal.copy(), 'position': raw,
           'elbow_error_ticks': goal[2] - raw[2],
           'elbow_displacement_ticks': raw[2] - trial['start_raw'][2]}
    # Keep even the triggering or malformed sample in the return package.
    trial['samples'].append(row)
    limiter.last = goal.copy()
    limiter.check_measured(raw, enforce_tracking=True)
    for i in (0, 1, 3):
        if abs(raw[i] - initial[i]) * DEG_PER_TICK > 2:
            raise RuntimeError(f'{MOTORS[i]} moved more than 2 degrees')
    if abs(raw[5] - initial[5]) > 15:
        raise RuntimeError('Gripper moved')
    for name in ('Torque_Enable', 'Goal_Position', 'Present_Load', 'Present_Current', 'Present_Voltage', 'Status'):
        row[name] = arm.bus.sync_read(name, normalize=False, num_retry=0)
    if [int(row['Torque_Enable'][n]) for n in MOTORS] != EXPECTED:
        arm.force_release = True
        raise RuntimeError('Unexpected torque pattern')
    if any(int(v) != 0 for v in row['Status'].values()):
        raise RuntimeError('Motor status nonzero')
    if any(int(row['Goal_Position'][n]) != goal[i] for i, n in enumerate(MOTORS) if i != 4):
        raise RuntimeError('Goal readback mismatch')
    return raw, row


def run(arm, cal, report, clock=time.perf_counter, sleep=time.sleep, stopped=stop_requested):
    goal = None
    try:
        arm.connect()
        arm.verify('Torque_Enable', OFF)
        initial = arm.raw_read()
        report['initial_raw'] = initial
        limiter = GraspLimiter(cal, initial, freeze_wrist=True)
        if min(initial[2] - cal['elbow_flex']['range_min'], cal['elbow_flex']['range_max'] - initial[2]) < 48:
            raise RuntimeError('Elbow needs at least 48 ticks to both saved limits')
        sleep(.05)
        if stopped():
            raise InterruptedError('Stop requested before activation')
        second = arm.raw_read()
        report['preflight_second_raw'] = second
        report['preflight_delta_ticks'] = {n: second[i]-initial[i] for i, n in enumerate(MOTORS)}
        if max(abs(a-b) for a, b in zip(initial, second)) > 2:
            raise RuntimeError('Unstable start: ' + repr(report['preflight_delta_ticks']))
        goal = arm.enable_aligned(initial, stopped).copy()
        report['aligned_raw'] = goal.copy()
        for direction in (1, -1):
            arm.verify('P_Coefficient', [16]*6)
            start = arm.raw_read()
            if max(abs(start[i] - goal[i]) for i in (0, 1, 2, 3, 5)) > 2:
                raise RuntimeError('Position not settled before direction start')
            trial = {'direction': direction, 'start_raw': start, 'samples': [], 'status': 'running'}
            report['trials'].append(trial)
            begin = clock()
            previous_cycle = None
            while True:
                cycle = clock()
                if previous_cycle is not None and cycle - previous_cycle > .15:
                    raise TimeoutError('Cycle interval exceeded 150ms')
                previous_cycle = cycle
                if stopped():
                    raise InterruptedError('Stop requested')
                if cycle - begin > 8:
                    raise TimeoutError('Direction exceeded 8 seconds')
                raw, row = sample(arm, limiter, initial, goal, trial, 'before_step', clock)
                displacement = raw[2] - start[2]
                error = goal[2] - raw[2]
                if abs(displacement) >= 3:
                    trial['status'] = 'started' if direction * displacement > 0 else 'wrong_direction'
                    trial['start_error_ticks'] = error
                    trial['start_error_degrees'] = error * DEG_PER_TICK
                    if trial['status'] == 'wrong_direction':
                        raise RuntimeError('Elbow moved in opposite direction')
                    break
                if abs(error) >= 40:
                    trial['status'] = 'not_observed_within_40_ticks'
                    trial['start_error_ticks'] = None
                    break
                if clock() - cycle > .15:
                    raise TimeoutError('Read cycle exceeded 150ms; no further goal sent')
                if stopped():
                    raise InterruptedError('Stop requested before next goal')
                next_goal = goal.copy()
                next_goal[2] += direction
                if not cal['elbow_flex']['range_min'] <= next_goal[2] <= cal['elbow_flex']['range_max']:
                    raise RuntimeError('Next target exceeds saved elbow range')
                # A failed or partial transmission leaves the hardware state unknown.
                arm.write_register('Goal_Position', next_goal)
                goal = next_goal
                row['sent_target'] = goal.copy()
                elapsed = clock() - cycle
                if elapsed > .15:
                    raise TimeoutError('Cycle exceeded 150ms')
                sleep(max(0, .1 - elapsed))
            # Stop increasing the target; align active goals to the current reading.
            # The wrist is excluded by BoundedArm's serial whitelist.
            hold = arm.hold()
            if not hold['all_expected_torque_states']:
                raise RuntimeError('Unexpected torque after hold')
            goal = hold['raw_goal'].copy()
            trial['hold'] = hold
            sleep(.2)
            settled, _ = sample(arm, limiter, initial, goal, trial, 'hold_check', clock)
            if max(abs(settled[i]-goal[i]) for i in (0, 1, 2, 3, 5)) > 2:
                raise RuntimeError('Position not settled after hold; no next direction')
        report['status'] = 'completed'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
    finally:
        if arm.may_be_enabled:
            try:
                # Operator must support arm throughout: torque-off can let it fall.
                arm.release()
                report['release_confirmed'] = True
            except BaseException:
                report['release_error'] = traceback.format_exc()
                report['status'] = 'release_not_confirmed'
        try:
            arm.close()
        except BaseException:
            report['close_error'] = traceback.format_exc()
            report['status'] = 'close_not_confirmed'
        report['bus'] = arm.report()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not args.run:
        raise SystemExit('Explicit --run required')
    out = HERE / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f_ElbowThreshold')
    out.mkdir(parents=True)
    report = {'status': 'starting', 'trials': [], 'limits': {'step_ticks': 1, 'period_ms': 100,
              'movement_ticks': 3, 'error_ticks': 40, 'tracking_degrees': 4},
              'note': 'Raw load/current values are not physical torque. Position goal may enable torque.'}
    sent = []
    try:
        cal = validate_calibration(CAL_PATH, HERE/'reference_calibration.json')
        arm = BoundedArm('COM5', cal, True, freeze_wrist=True)
        original = arm.guard.write_port
        def trace(packet):
            data = bytes(packet)
            if len(data) > 4 and data[4] not in (1, 2, 0x82):
                sent.append(data.hex())
            return original(packet)
        arm.guard.write_port = trace
        run(arm, cal, report)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
    report['write_frames_hex'] = sent
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = out.with_name(out.name+'_return.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out/'summary.json', 'summary.json')
    print('STATUS:', report['status'])
    for t in report['trials']:
        print('DIRECTION:', t['direction'], 'RESULT:', t['status'], 'START_ERROR_TICKS:', t.get('start_error_ticks'))
    if report.get('error'):
        print(report['error'])
    print('RELEASE CONFIRMED:', report.get('release_confirmed', False))
    print('SEND BACK:', archive)
    if report['status'] != 'completed':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
