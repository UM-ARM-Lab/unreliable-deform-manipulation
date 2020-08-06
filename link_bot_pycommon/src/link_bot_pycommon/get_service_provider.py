from link_bot_gazebo_python.gazebo_services import GazeboServices
from victor.victor_services import VictorServices


def get_service_provider(service_provider_name):
    if service_provider_name == 'victor':
        return VictorServices()
    elif service_provider_name == 'gazebo':
        return GazeboServices()
    else:
        raise NotImplementedError()
