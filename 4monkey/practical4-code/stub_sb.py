import numpy.random as npr
import numpy as np
import sys
import random

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self, ds, alpha, gamma, eps, init, buckets):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        self.gamma = gamma        
        self.alpha = alpha
        self.eps = eps
        self.init = init
        self.buckets = buckets
        self.numIters = 0
        self.numEpochs = 0
        self.max_score = 0
        
        self.act_list = [0, 1]
        
        # Discrete State Array that indexes Q
        self.ds = ds

        # Q function
        self.Q = [[self.init for a in range(len(self.act_list))] for s in range(self.buckets**6)]

    def discreteState(self, state):
        #velocity = [-50, 30]
        ds = []
        sw = 600.
        sh = 450.
        
        #ds.append(int((state['tree']['dist'])*(self.buckets-1)/sw + 1))
        ds.append(int((state['tree']['top'])*(self.buckets-1)/sh + 1))
        ds.append(int((state['tree']['bot'])*(self.buckets-1)/sh + 1))
        #ds.append(int(((state['monkey']['vel'])+50)/(80/self.buckets)))
        ds.append(int((state['monkey']['top'])*(self.buckets-1)/sh + 1))
        ds.append(int((state['monkey']['bot'])*(self.buckets-1)/sh + 1))
        
        #print ds
        ret = 0
        
        for i in range(len(ds)):
            ret += ds[i]*(self.buckets**i)
            
        #print ret
        return ret

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.numEpochs += 1

    def chooseAction(self, state, Q):
        best_action = 0
        best_Q = -float("inf")
        # In the case of epsilon, return a random action
        if (random.random() < self.eps(self.numIters+1)):
            action = random.choice(self.act_list)
            return action
        else:
            # Shuffle the action list so that we select new
            # actions in case of ties. Then maximize
            random.shuffle(self.act_list)
            for a in self.act_list:
                #print state, len(Q)
                temp_Q = Q[state][a]
                #print temp_Q
                if temp_Q > best_Q:
                    best_Q = temp_Q
                    best_action = a
            #print best_action
            #print "best", best_Q
            return best_action
        
    # Function to choose function based on max value
    def action_callback(self, state):
        if not self.last_state:
            self.last_state = self.discreteState(state)
            self.last_action = self.chooseAction(self.last_state, self.Q)
        else:
            # Choose A from S using policy derived from Q
            action = self.last_action     

            # Choose A' from S' using policy derived from Q
            new_state  = self.discreteState(state)
            new_action = self.chooseAction(new_state, self.Q)

            # SARSA
            self.Q[self.last_state][action] = self.Q[self.last_state][action] \
                                                   + self.alpha(self.numEpochs+1)*(self.last_reward + \
                                                                 self.gamma*self.Q[new_state][new_action] - \
                                                                 self.Q[self.last_state][action])
                
            # Update all vals
            self.last_action = new_action
            self.last_state  = new_state
            
        self.numIters += 1
        return self.last_action

        

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if self.last_reward:
            self.last_reward += reward
        else:
            self.last_reward = reward
        #print self.last_reward

iters = 100
epsilon = 0.15
epsilon = lambda(x): 1./x
gamma = .95
alpha = .1
ds = []

            
learner = Learner(ds, lambda(x): min(.1, 1./x), gamma, epsilon, 0, 4)

for ii in xrange(iters):
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass
    if swing.get_state()['score'] > learner.max_score:
        learner.max_score = swing.get_state()['score']
    # Reset the state of the learner.
    learner.reset()

print learner.max_score



    
