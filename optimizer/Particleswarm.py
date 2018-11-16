from __future__ import division
import random
import math
from optimizer.optimizer_interface import optimizer_interface
import numpy as np


class Particle:
    """
    To simulate one particle : evalue,update_vel,update_pos,evaluate,update_vel,update_pos etc...
    """
    def __init__(self, vec_init_pos, scale_init_vel, momentum, PI_my_vel, PI_social_vel):
        self.current_pos = vec_init_pos
        self.current_loss = None
        self.current_velocity = np.random.normal(0, scale_init_vel, vec_init_pos.shape)
        self.my_best_pos = np.copy(self.current_pos)
        self.my_best_loss = None

        self.momentum = momentum
        self.PI_my_vel = PI_my_vel
        self.PI_social_vel = PI_social_vel

    def __save_current_as_best(self):
        self.my_best_pos = self.current_pos
        self.my_best_loss = self.current_loss

    def evaluate_and_save_local_best(self, costFunc):
        self.current_loss = costFunc(self.current_pos)

        # test if the current position is its best pos
        if self.my_best_loss is None:
            self.__save_current_as_best()
        elif self.current_loss < self.my_best_loss :
            self.__save_current_as_best()

    def update_velocity(self, GLOBAL_BEST_POSITION):
        vel_cognitive = 1 * self.PI_my_vel() * (self.my_best_pos - self.current_pos)
        vel_social = 2 * self.PI_social_vel() * (GLOBAL_BEST_POSITION - self.current_pos)
        self.current_velocity = self.momentum * self.current_velocity + vel_cognitive + vel_social
    def update_velocity0(self, GLOBAL_BEST_POSITION):
        for i in range(0, self.current_pos.shape[0]): # we loop to save memory
            rand1 = random.random()
            rand2 = random.random()
            vel_cognitive = 1 * rand1 * (self.my_best_pos[i] - self.current_pos[i])
            vel_social = 2 * rand2 * (GLOBAL_BEST_POSITION[i] - self.current_pos[i])
            self.current_velocity[i] = self.momentum * self.current_velocity[i] + vel_cognitive + vel_social



    def collide(self, scale_min, scale_max):
        for i in range(0, self.current_pos.shape[0]):
            if self.current_pos[i] > scale_max:
                self.current_pos[i] = scale_max
            if self.current_pos[i] < scale_min:
                self.current_pos[i] = scale_min

    def update_position(self):
        self.current_pos = self.current_pos + self.current_velocity



class Swarm(optimizer_interface):
    def __init__(self,num_particles,init_x_swarm, scale_init_vel,apriori_nn,momentum=0.5,my_vel_contrib=1.,social_vel_contrib=2.):
        optimizer_interface.__init__(self)
        self.num_particles=num_particles
        self.swarm=[]

        w0=np.zeros(init_x_swarm.shape)
        def PI_my_vel():
            return np.random.uniform(size=w0.shape)
            #return np.abs(apriori_nn(w0))/2.
            #return np.abs(np.random.normal(size=w0.shape))/2.
        def PI_social_vel():
            return np.random.uniform(size=w0.shape)
            #return np.abs(apriori_nn(w0)) / 2.
            #return np.abs(np.random.normal(size=w0.shape))/2.

        for i in range(self.num_particles):
            p=Particle(vec_init_pos=init_x_swarm,
                       scale_init_vel=scale_init_vel,
                       momentum=momentum,
                       PI_my_vel=PI_my_vel,
                       PI_social_vel=PI_social_vel)
            self.swarm.append(p)

    def get_best_particle(self):
        best_i=0
        for i in range(0, self.num_particles):
            if self.swarm[i].my_best_loss < self.swarm[best_i].my_best_loss:
                best_i=i
        return self.swarm[best_i]

    def run_one_step(self,x,function_to_min):
        # 1. update all particles loss and save as best if it is the best position ever found
        for i in range(0, self.num_particles):
            self.swarm[i].evaluate_and_save_local_best(function_to_min.f)

        # 2. compute current best pos of the best particle
        best_particle=self.get_best_particle()

        # 3. move particles
        for i in range(0, self.num_particles):
            self.swarm[i].update_velocity(GLOBAL_BEST_POSITION=best_particle.my_best_pos)
            self.swarm[i].update_position()
            self.swarm[i].collide(-2, +2)

        return best_particle.my_best_pos