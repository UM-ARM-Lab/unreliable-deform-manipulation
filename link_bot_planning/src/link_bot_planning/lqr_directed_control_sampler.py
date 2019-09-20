import numpy as np

from link_bot_planning.my_directed_control_sampler import MyDirectedControlSampler


class LQRDirectedControlSampler(MyDirectedControlSampler):

    def __init__(self, si, lqr_solver):
        super(LQRDirectedControlSampler, self).__init__(si, lqr_solver, "LQR")

    def sampleTo(self, control_out, previous_control, state, target_out):
        # we return 0 to indicate no duration when LQR gives us a control that takes us into collision
        # this will cause the RRT to throw out this motion
        M = self.si.getStateSpace().getSubspace(0).getDimension()
        P = self.si.getStateSpace().getSubspace(1).getDimension()
        L = self.si.getControlSpace().getDimension()
        o_d = np.ndarray((M, 1))
        o_d_target = np.ndarray((M, 1))
        o_k = np.ndarray((P, 1))
        for i in range(M):
            o_d[i, 0] = state[0][i]
            o_d_target[i, 0] = target_out[0][i]
        for i in range(P):
            o_k[i, 0] = state[1][i]

        u, o_d_next, o_k_next = self.action_selector.dual_act(o_d, o_k, o_d_target)

        for i in range(L):
            control_out[i] = u[0, 0, i]
        for i in range(M):
            target_out[0][i] = o_d_next[i, 0]
        for i in range(P):
            target_out[1][i] = o_k_next[i, 0]

        LQRDirectedControlSampler.states_sampled_at.append(state)

        # check validity
        if not self.si.isValid(target_out):
            return 0

        duration_steps = 1
        return duration_steps

    def dual_shortcut(self, o_d, o_k, o_d_goal):
        us = []
        o_ks = [o_k]
        o_ds = [o_d]
        errors = []
        for _ in range(100):
            u, o_d_next, o_k_next = self.action_selector.dual_act(o_d, o_k, o_d_goal)
            constraint_violated = self.action_selector.constraint_violated(o_k_next)

            if constraint_violated:
                return False, None, None, None
            if np.linalg.norm(u) < 1e-2:
                return False, None, None, None

            us.append(u)
            o_ds.append(o_d)
            o_ks.append(o_k)

            error = np.linalg.norm(o_d - o_d_goal)
            errors.append(error)
            if error < 0.05:
                return True, np.array(us), np.array(o_ds), np.array(o_ks)
            o_d = o_d_next
            o_k = o_k_next

        return False, None, None, None
