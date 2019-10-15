import numpy as np
import ompl.util as ou
from ompl import base as ob
from ompl import control as oc

from link_bot_gazebo.gazebo_utils import GazeboServices, get_local_occupancy_data
from link_bot_planning.my_motion_validator import MotionClassifier
from link_bot_planning.params import LocalEnvParams
from link_bot_planning.state_spaces import to_numpy, from_numpy
from link_bot_pycommon import link_bot_sdf_utils


class ShootingDirectedControlSampler(oc.DirectedControlSampler):

    def __init__(self,
                 si: ob.StateSpace,
                 fwd_model,
                 classifier_model: MotionClassifier,
                 services: GazeboServices,
                 local_env_params: LocalEnvParams,
                 max_v: float,
                 n_samples: int):
        super(ShootingDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'shooting_dcs'
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.n_samples = n_samples
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.services = services
        self.local_env_params = local_env_params
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getSubspace(0).getDimension()
        self.n_local_env = self.state_space.getSubspace(1).getDimension()
        self.n_control = self.control_space.getDimension()
        self.internal = ShootingDirectedControlSamplerInternal(self.fwd_model,
                                                               self.classifier_model,
                                                               self.services,
                                                               self.local_env_params,
                                                               self.max_v,
                                                               self.n_samples,
                                                               self.n_state,
                                                               self.n_local_env)

    @classmethod
    def alloc(cls,
              si: ob.StateSpace,
              fwd_model,
              classifier_model: MotionClassifier,
              services: GazeboServices,
              local_env_params: LocalEnvParams,
              max_v: float,
              n_samples: int):
        return cls(si, fwd_model, classifier_model, services, local_env_params, max_v, n_samples)

    @classmethod
    def allocator(cls,
                  fwd_model,
                  classifier_model: MotionClassifier,
                  services: GazeboServices,
                  local_env_params: LocalEnvParams,
                  max_v: float,
                  n_samples: int = 10):
        def partial(si: ob.StateSpace):
            return cls.alloc(si, fwd_model, classifier_model, services, local_env_params, max_v, n_samples)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self,
                 control_out,  # TODO: how to annotate the type of this?
                 previous_control,
                 state: ob.CompoundStateInternal,
                 target_out: ob.CompoundStateInternal):
        np_s = to_numpy(state[0], self.n_state)
        np_target = to_numpy(target_out[0], self.n_state)
        np_s_reached, u, local_env_data = self.internal.sampleTo(np_s, np_target)

        from_numpy(u, control_out, self.n_control)
        from_numpy(np_s_reached, target_out[0], self.n_state)
        local_env = local_env_data.data.flatten().astype(np.float64)
        for idx in range(self.n_local_env):
            occupancy_value = local_env[idx]
            target_out[1][idx] = occupancy_value
        origin = local_env_data.origin.astype(np.float64)
        from_numpy(origin, target_out[2], 2)

        # check validity
        duration_steps = 1
        if not self.si.isValid(target_out):
            duration_steps = 0

        return duration_steps


class ShootingDirectedControlSamplerInternal:
    states_sampled_at = []

    def __init__(self,
                 fwd_model,
                 classifier_model: MotionClassifier,
                 services: GazeboServices,
                 local_env_params: LocalEnvParams,
                 max_v: float,
                 n_samples: int,
                 n_state: int,
                 n_local_env: int):
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.n_samples = n_samples
        self.fwd_model = fwd_model
        self.classifier_model = classifier_model
        self.services = services
        self.local_env_params = local_env_params
        self.n_state = n_state
        self.n_local_env = n_local_env

    def sampleTo(self,
                 state: np.ndarray,
                 target: np.ndarray) -> [np.ndarray, np.ndarray, link_bot_sdf_utils.OccupancyData]:
        self.states_sampled_at.append(target)

        head_point = state[0, 4:6]
        local_env_data = get_local_occupancy_data(cols=self.local_env_params.w_cols,
                                                        rows=self.local_env_params.h_rows,
                                                        res=self.local_env_params.res,
                                                        center_point=head_point,
                                                        services=self.services)

        min_distance = np.inf
        min_u = None
        min_np_s_next = None
        for i in range(self.n_samples):
            # sample a random action
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.array([[self.max_v * np.cos(theta), self.max_v * np.sin(theta)]])
            batch_u = np.expand_dims(u, axis=0)

            # use the forward model to predict the next configuration
            points_next = self.fwd_model.predict(state, batch_u)
            np_s_next = points_next[:, 1].reshape([1, self.n_state])

            # check that the motion is valid
            accept_probability = self.classifier_model.predict(local_env_data, state, np_s_next)
            if self.rng_.uniform01() > accept_probability:
                # reject
                continue

            # keep if it's the best we've seen
            distance = np.linalg.norm(np_s_next - target)
            if distance < min_distance:
                min_distance = distance
                min_u = u
                min_np_s_next = np_s_next

        return min_np_s_next, min_u, local_env_data
