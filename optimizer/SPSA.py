import numpy as np

from optimizer import Interface_optimizer

nb = 100
class SPSA(Interface_optimizer):
    def __init__(self, lr, epsilon_length, PI=None):
        Interface_optimizer.__init__(self)
        self.lr = lr
        self.epsilon_length = epsilon_length

        if PI is None:
            def PI(x):
                return np.random.normal(x, .1)
        self.PI = PI


    def run_one_step(self, x, function_to_min):
        fx = function_to_min.f(x)
        random_positive_vector_norm = np.abs(self.PI(np.zeros(x.shape))) / 2.
        random_positive_vector_norm = (random_positive_vector_norm / np.linalg.norm(random_positive_vector_norm))
        random_positive_vector_norm = self.epsilon_length * random_positive_vector_norm

        fx_proposed = function_to_min.f(x + random_positive_vector_norm)
        coef_direction = (fx_proposed - fx) / (self.epsilon_length)  # 0> if it's better

        grad = self.lr * coef_direction * random_positive_vector_norm

        new_x = x - grad

        return new_x

    def run_one_step0(self, x, function_to_min):
        # fx=function_to_min.f(x)

        # compute delta value
        x_epsilon = np.abs(self.PI(np.zeros(x.shape))) / 2.
        x_epsilon = (x_epsilon / np.linalg.norm(x_epsilon))
        x_epsilon = self.epsilon_length * x_epsilon

        fxa_proposed = function_to_min.f(x + x_epsilon)
        fxb_proposed = function_to_min.f(x - x_epsilon)

        coef_direction = (fxa_proposed - fxb_proposed) / (2 * self.epsilon_length)

        new_x = x - (self.lr * coef_direction * x_epsilon)

        return new_x
