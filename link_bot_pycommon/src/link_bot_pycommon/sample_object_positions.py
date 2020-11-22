from typing import Dict

from geometry_msgs.msg import Vector3


def sample_object_position(env_rng, xyz_range: Dict):
    x_range = xyz_range['x']
    y_range = xyz_range['y']
    z_range = xyz_range['z']
    position = Vector3()
    position.x = env_rng.uniform(*x_range)
    position.y = env_rng.uniform(*y_range)
    position.z = env_rng.uniform(*z_range)
    return position


def sample_object_positions(env_rng, movable_objects: Dict) -> Dict[str, Dict]:
    random_object_positions = {name: sample_object_position(
        env_rng, xyz_range) for name, xyz_range in movable_objects.items()}
    return random_object_positions