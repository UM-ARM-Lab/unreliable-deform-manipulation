class ActionSelector(object):

    def __init__(self):
        pass

    def just_d_act(self, o_d, o_d_goal):
        raise NotImplementedError("ActionSelector is an abstract class.")

    def dual_act(self, o_d, o_k, o_d_goal):
        raise NotImplementedError("ActionSelector is an abstract class.")

