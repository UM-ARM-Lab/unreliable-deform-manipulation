import ompl.util as ou
from ompl import control as oc
from ompl import base as ob
import numpy as np

from link_bot_gazebo.gazebo_utils import get_local_sdf_data, GazeboServices
from link_bot_planning.my_motion_validator import MotionClassifier
from link_bot_planning.shooting_rrt_mpc import SDFParams
from link_bot_planning.state_spaces import to_numpy, from_numpy


class ShootingDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self,
                 si: ob.StateSpace,
                 fwd_model,
                 classifier_model: MotionClassifier,
                 services: GazeboServices,
                 sdf_params: SDFParams,
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
        self.sdf_params = sdf_params
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getDimension()
        self.n_control = self.control_space.getDimension()
        self.min_steps = int(self.si.getMinControlDuration())
        self.max_steps = int(self.si.getMaxControlDuration())

    @classmethod
    def alloc(cls,
              si: ob.StateSpace,
              fwd_model,
              classifier_model: MotionClassifier,
              services: GazeboServices,
              sdf_params: SDFParams,
              max_v: float,
              n_samples: int):
        return cls(si, fwd_model, classifier_model, services, sdf_params, max_v, n_samples)

    @classmethod
    def allocator(cls,
                  fwd_model,
                  classifier_model: MotionClassifier,
                  services: GazeboServices,
                  sdf_params: SDFParams,
                  max_v: float,
                  n_samples: int = 10):
        def partial(si: ob.StateSpace):
            return cls.alloc(si, fwd_model, classifier_model, services, sdf_params, max_v, n_samples)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self, control_out, previous_control, state, target_out):
        np_s = to_numpy(state, self.n_state)
        np_target = to_numpy(target_out, self.n_state)

        self.states_sampled_at.append(np_target)

        min_distance = np.inf
        min_u = None
        min_np_s_next = None
        for i in range(self.n_samples):
            # sample a random action
            theta = np.random.uniform(-np.pi, np.pi)
            u = np.array([[self.max_v * np.cos(theta), self.max_v * np.sin(theta)]])
            batch_u = np.expand_dims(u, axis=0)

            # use the forward model to predict the next configuration
            head_point = np_s[0, 4:5]
            points_next = self.fwd_model.predict(np_s, batch_u)
            np_s_next = points_next[:, 1].reshape([1, self.n_state])

            # check that the motion is valid
            local_sdf_data = get_local_sdf_data(sdf_cols=self.sdf_params.local_w_cols,
                                                sdf_rows=self.sdf_params.local_h_rows,
                                                res=self.sdf_params.res,
                                                origin_point=head_point,
                                                services=self.services)

            accept_probability = self.classifier_model.predict(local_sdf_data, np_s, np_s_next)
            if self.rng_.uniform01() >= accept_probability:
                continue

            # keep if it's the best we've seen
            distance = np.linalg.norm(np_s_next - np_target)
            if distance < min_distance:
                min_distance = distance
                min_u = u
                min_np_s_next = np_s_next

        from_numpy(min_u, control_out, self.n_control)
        from_numpy(min_np_s_next, target_out, self.n_state)

        # check validity
        duration_steps = 1
        if not self.si.isValid(target_out):
            duration_steps = 0

        return duration_steps
