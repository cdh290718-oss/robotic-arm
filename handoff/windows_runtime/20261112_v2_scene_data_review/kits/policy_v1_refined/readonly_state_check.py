"""Read-only SO101 state check: torque / P / D / Lock / present position. No writes.

Purpose: before any execute run, confirm current torque state and controller
gains without sending a single write packet. Write packets are blocked at the
serial TX boundary by ReadOnlyPacketGuard.
"""
import json
import sys
from pathlib import Path
from readonly_bus import MOTORS, ReadOnlyArm, validate_calibration

HERE = Path(__file__).resolve().parent
PORT = 'COM5'
CAL_PATH = Path('G:/LeRobot/data/calibration/robots/so_follower/lerobot.json')


def main():
    calibration = validate_calibration(CAL_PATH, HERE/'reference_calibration.json')
    arm = ReadOnlyArm(PORT, calibration)
    arm.connect()
    torque = arm.bus.sync_read('Torque_Enable', normalize=False)
    pos = arm.bus.sync_read('Present_Position')
    p = arm.bus.sync_read('P_Coefficient', normalize=False)
    d = arm.bus.sync_read('D_Coefficient', normalize=False)
    lock = arm.bus.sync_read('Lock', normalize=False)
    print(json.dumps({'torque_enable': torque, 'present': pos, 'P': p, 'D': d, 'Lock': lock},
                     ensure_ascii=False, indent=2, sort_keys=True))
    arm.close()
    print(json.dumps({'bus_report': arm.report()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
