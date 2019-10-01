import ompl.util as ou
from ompl import control as oc

from link_bot_planning.state_spaces import to_numpy, from_numpy


class GPDirectedControlSampler(oc.DirectedControlSampler):
    states_sampled_at = []

    def __init__(self, si, fwd_gp_model, inv_gp_model, max_v):
        super(GPDirectedControlSampler, self).__init__(si)
        self.si = si
        self.name_ = 'gp_dcs'
        self.rng_ = ou.RNG()
        self.max_v = max_v
        self.fwd_gp_model = fwd_gp_model
        self.inv_gp_model = inv_gp_model
        self.state_space = self.si.getStateSpace()
        self.control_space = self.si.getControlSpace()
        self.n_state = self.state_space.getDimension()
        self.n_control = self.control_space.getDimension()
        self.min_steps = int(self.si.getMinControlDuration())
        self.max_steps = int(self.si.getMaxControlDuration())
        self.fwd_gp_model.initialize_rng(self.min_steps, self.max_steps)
        if inv_gp_model is not None:
            self.inv_gp_model.initialize_rng(self.min_steps, self.max_steps)

    @classmethod
    def alloc(cls, si, fwd_gp_model, inv_gp_model, max_v):
        return cls(si, fwd_gp_model, inv_gp_model, max_v)

    @classmethod
    def allocator(cls, fwd_gp_model, inv_gp_model, max_v):
        def partial(si):
            return cls.alloc(si, fwd_gp_model, inv_gp_model, max_v)

        return oc.DirectedControlSamplerAllocator(partial)

    def sampleTo(self, control_out, previous_control, state, target_out):
        np_s = to_numpy(state, self.n_state)
        np_target = to_numpy(target_out, self.n_state)

        # # construct a new np_target which is at most 1 meter away from the rope
        # # 1 meter is ~80th percentile on how far the head moved in the training data for the inverse GP
        # np_s_tail = np_s.reshape(-1, 2)[0]
        # np_target_pts = np_target.reshape(-1, 2)
        # np_target_tail = np_target_pts[0]
        # tail_delta = np_target_tail - np_s_tail
        # max_tail_delta = 1.0
        # new_tail = np_s_tail
        # if np.linalg.norm(tail_delta) > max_tail_delta:
        #     new_tail = np_s_tail + tail_delta / np.linalg.norm(tail_delta) * max_tail_delta
        # new_np_target = (np_target_pts - np_target_tail + new_tail).reshape(-1, self.n_state)
        # np_target = new_np_target

        self.states_sampled_at.append(np_target)

        if self.inv_gp_model is None:
            u = self.fwd_gp_model.dumb_inv_act(np_s, np_target, self.max_v)
        else:
            u = self.inv_gp_model.inv_act(np_s, np_target, self.max_v)

        np_s_next = self.fwd_gp_model.fwd_act(np_s, u)

        from_numpy(u, control_out, self.n_control)
        from_numpy(np_s_next, target_out, self.n_state)

        # check validity
        duration_steps = 1
        if not self.si.isValid(target_out):
            duration_steps = 0

        return duration_steps
