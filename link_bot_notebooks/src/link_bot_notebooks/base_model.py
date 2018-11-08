class BaseModel:

    def __init__(self, N, M, L):
        """
        N: dimensionality of the full state
        M: dimensionality in the reduced state
        L: dimensionality in the actions
        """
        self.N = N
        self.M = M
        self.L = L

    def size(self):
        pass

    def reduce(self, s):
        pass

    def predict(self, o, u):
        pass

    def cost(self, o, g):
        pass

    def save(self, outfile):
        pass

    def load(self, infile):
        pass

    def __repr__(self):
        pass


class ModelWrapper:

    def __init__(self, model):
        self.model = model

    def predict_from_s(self, s, u):
        return self.model.predict(self.model.reduce(s), u)

    def predict_from_o(self, o, u):
        return self.model.predict(o, u)

    def cost_of_s(self, s, g):
        return self.model.cost(self.model.reduce(s), g)

    def predict_cost_of_s(self, s, u, g):
        return self.model.cost(self.model.predict(self.model.reduce(s), u), g)

    def predict_cost(self, o, u, g):
        return self.model.cost(self.model.predict(o, u), g)

