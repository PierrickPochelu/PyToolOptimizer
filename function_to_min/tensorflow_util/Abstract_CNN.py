import tensorflow as tf
import numpy as np

class CNN():
    def __init__(self):
        self.forward_model=None
        self.loss_model=None
        self.placeholder_x=None
        self.placeholder_y=None

    def get_forward_model(self):
        return self.forward_model

    def get_loss_model(self):
        return self.loss_model

    def  get_placeholder_x(self):
        return self.placeholder_x

    def get_placeholder_y(self):
        return self.placeholder_y