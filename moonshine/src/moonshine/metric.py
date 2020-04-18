class Metric:

    @staticmethod
    def is_better_than(a, b):
        raise NotImplementedError()

    @staticmethod
    def key():
        raise NotImplementedError()


class LossMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        return a < b

    @staticmethod
    def key():
        return "loss"


class AccuracyMetric(Metric):

    @staticmethod
    def is_better_than(a, b):
        return a > b

    @staticmethod
    def key():
        return "accuracy"
