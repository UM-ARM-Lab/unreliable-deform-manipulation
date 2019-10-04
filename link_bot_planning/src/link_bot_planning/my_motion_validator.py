import ompl.base as ob


class MyMotionValidator(ob.MotionValidator):

    def __init__(self, si, validator_model):
        super(MyMotionValidator, self).__init__(si)

    def checkMotion(self, s1, s2):
        return True
