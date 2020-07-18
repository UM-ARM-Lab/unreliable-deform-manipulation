#!/usr/bin/env python
import json
import argparse
import pathlib

from link_bot_pycommon.dual_floating_gripper_scenario import DualFloatingGripperRopeScenario
from link_bot_pycommon.ros_pycommon import get_environment_for_extents_3d
from link_bot_pycommon.serialization import dummy_proof_write
from link_bot_gazebo_python.gazebo_services import GazeboServices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recovery_dataset_dir", type=pathlib.Path,
                        help='the hparams.json file for the recovery dataset')

    args = parser.parse_args()
    scenario = DualFloatingGripperRopeScenario()
    service_provider = GazeboServices()
    state = scenario.get_state()

    with args.recovery_dataset_dir.open("r") as hparams_file:
        hparams = json.load(hparams_file)
    data_collection_params = hparams['data_collection_params']
    environment = get_environment_for_extents_3d(extent=data_collection_params['extent'],
                                                 res=data_collection_params['res'],
                                                 service_provider=service_provider,
                                                 robot_name=scenario.robot_name())

    data = {
        'state': state,
        'environment': environment
    }
    dummy_proof_write


if __name__ == "__main__":
    main()
