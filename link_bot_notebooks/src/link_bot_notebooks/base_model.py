class BaseModel:

    def __init__(self, N, M, L, P=0):
        """
        N: dimensionality of the full state
        M: dimensionality in the reduced state
        L: dimensionality in the actions
        P: dimensionality in the constraints
        """
        self.N = N
        self.M = M
        self.L = L
        self.P = P

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

    def predict_from_s(self, s, u):
        return self.predict(self.reduce(s), u)

    def predict_from_o(self, o, u):
        return self.predict(o, u)

    def cost_of_s(self, s, g):
        return self.cost(self.reduce(s), g)

    def predict_cost_of_s(self, s, u, g):
        return self.cost(self.predict(self.reduce(s), u), g)

    def predict_cost(self, o, u, g):
        return self.cost(self.predict(o, u), g)

    def __repr__(self):
        pass
