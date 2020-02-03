class VizObject:

    def __init__(self):
        self.states_sampled_at = []
        self.rejected_samples = []
        self.debugging1 = None
        self.debugging2 = None
        self.new_sample = False

    def clear(self):
        self.states_sampled_at.clear()
        self.rejected_samples.clear()