class VizObject:

    def __init__(self):
        self.states_sampled_at = []
        self.rejected_samples = []

    def clear(self):
        self.states_sampled_at.clear()
        self.rejected_samples.clear()