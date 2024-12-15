class GoalGradient():
    def __init__(self, A, Y=None):
        self.A = A
        self.Y = Y

    def xtAx(self, x):
        return x.T.dot(self.A.matvec(x))

    def xtx(self, x):
        return x.T.dot(x)

    def objective_function(self, x, lambd):
        xtAx = self.xtAx(x)
        xtx = self.xtx(x)
        if lambd is not None:
            lambYT = lambd.T.dot(self.Y.T.dot(x))
            return xtAx/xtx + lambYT, xtAx/xtx, lambYT
        else:
            return xtAx/xtx, xtAx/xtx, 0

    def gradient_x(self, x, lambd):
        Ax = self.A.matvec(x)
        xtAx = self.xtAx(x)
        xtx = self.xtx(x)
        term1 = 2/xtx * (Ax - (xtAx/xtx * x))
        if self.Y is not None:
            term2 = self.Y.dot(lambd)
            gradient = term1 + term2
        else:
            gradient = term1
        return gradient

    def gradient_lambda(self, x):
        if self.Y is not None:
            return self.Y.T.dot(x)

    def goal_gradient(self, x, lambd):
        return self.gradient_x(x, lambd), self.gradient_lambda(x)