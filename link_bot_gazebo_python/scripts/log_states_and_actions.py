#!/usr/bin/env python

import argparse
import json
import pathlib

import numpy as np
import rospy

from link_bot_gazebo import gazebo_services
from link_bot_pycommon.get_scenario import get_scenario
from link_bot_planning.plan_and_execute import get_environment_common
from link_bot_pycommon.args import my_formatter
from moonshine.moonshine_utils import listify
from peter_msgs.msg import Action
from victor import victor_services


def main():
    rospy.init_node("log_states_and_actions")

    parser = argparse.ArgumentParser(formatter_class=my_formatter)
    parser.add_argument("scenario", choices=['link_bot', 'tether'])
    parser.add_argument('actions', type=pathlib.Path)
    parser.add_argument('outfile', type=pathlib.Path)
    parser.add_argument("--service-provider", choices=["gazebo", "victor"], default='gazebo')
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--w-m', type=float, default=2.0)
    parser.add_argument('--h-m', type=float, default=2.0)
    parser.add_argument('--res', type=float, default=0.01)
    parser.add_argument('--reset', action='store_true')

    args = parser.parse_args()

    # Start Services
    if args.service_provider == 'victor':
        service_provider = victor_services.VictorServices()
    else:
        service_provider = gazebo_services.GazeboServices()

    if args.reset:
        service_provider.reset_world(verbose=1, reset_robot=[0.51, 0])

    actions = np.genfromtxt(args.actions, delimiter=',')

    scenario = get_scenario(args.scenario)

    environment = get_environment_common(w_m=args.w_m,
                                         h_m=args.h_m,
                                         res=args.res,
                                         service_provider=service_provider,
                                         scenario=scenario)
    path = []
    for action in actions:
        state_t = {}
        action_request = Action()
        action_request.max_time_per_step = args.dt
        action_request.action = action
        action_response = scenario.execute_action(action_request)
        for named_object in action_response.objects.objects:
            state_t[named_object.name] = named_object.state_vector
        path.append(state_t)

    logged_data = {
        'environment': listify(environment),
        'path': path,
        'actions': listify(actions),
        'scenario': args.scenario,
    }

    json.dump(logged_data, args.outfile.open("w"), indent=2)


if __name__ == '__main__':
    main()
