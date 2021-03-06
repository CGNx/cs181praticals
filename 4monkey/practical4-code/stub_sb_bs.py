#import numpy.random as npr
import numpy as np
import sys
import random

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self, alpha, gamma, eps, init, buckets, Q = [], N = []):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        # Gamma = discount factor
        # Alpha = learning rate (function of some x)
        # Epsilon = greediness factor (function of some x)
        # Init = initial Q values
        # Buckets = discretization value
        self.gamma = gamma        
        self.alpha = alpha
        self.eps = eps
        self.init = init
        self.buckets = buckets
        self.Q = Q
        self.N = N

        # To keep track of where we are
        self.numIters = 0
        self.numEpochs = 0
        self.max_score = 0

        # List of actions
        self.act_list = [0, 1]
        if self.Q == []:
            # Q function, initialized to init over actions x states (2 x buckets^6)
            self.Q = [[self.init for a in range(len(self.act_list))] for s in range(10**2)]
            print self.buckets**3

        if self.N == []:
            # N function
            self.N = [[1 for a in range(len(self.act_list))] for s in range(10**2)]

    def discreteState(self, state):
        # Velocity range = [-50, 30]
        # Width and Height of Screen
        sw = 600.
        sh = 450.

        # Our Discrete State array
        ds = []

        # It seems that only top and bottom are valuable
        # Commenting out dist and vel literally put us
        # from maxing at 12 to maxing at 60
        
        # Scrapped velocity and distance
        #ds.append(int((state['tree']['dist'])*(self.buckets-1)/sw + 1))
        #ds.append(int(((state['monkey']['vel'])+50)/(80/self.buckets)))

        #ds.append(int((state['tree']['top'])*(self.buckets-1)/sh + 1))
        #ds.append(int((state['tree']['bot'])*(self.buckets-1)/sh + 1))

        #print state['tree']['top'], int(state['tree']['top']*(self.buckets-1)/(140) + 1)-7
        ds.append(int(state['tree']['top']*(self.buckets-1)/(sh-300) + 1)-7)
        #ds.append(int(state['tree']['top']*(self.buckets-1)/(140) + 1)-7)

        #ds.append(int((state['monkey']['top'])*(self.buckets-1)/sh + 1))
        #ds.append(int((state['monkey']['bot'])*(self.buckets-1)/sh + 1))
        
        monkeyAvg = (state['monkey']['top'] + state['monkey']['bot'])/2
        #print state['monkey']['top'], state['monkey']['bot'], monkeyAvg, int(monkeyAvg*(self.buckets-1)/(sh) + 1)
        # sh - (0, 450)
        ds.append(int(monkeyAvg*(self.buckets-1)/(sh) + 1))

        #ds = [treeTop, treeBot, monkTop, monkBot] -> [treeAvg, monkeyAvg]
        #ds = [7, 5, 8, 4] -> [0-3, 0-3]
        #ret = 7 + 5*6 + 8*36 + 4*216 -> 
        
        # Conver state to unique decimal representation
        ret = 0        
        for i in range(len(ds)):
            ret += ds[i]*(10**i)
        return ret

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        # Every time we reset, we enter a new Epoch
        self.numEpochs += 1

    def chooseAction(self, state, Q):
        best_action = 0
        best_Q = -float("inf")
        
        # In the case of epsilon, return a random action
        if (random.random() < self.eps(self.numEpochs+1)):
            action = random.choice(self.act_list)
            #if random.random() < .8:
                #return 0
            #else:
                #return 1
            return action
        
        else:
            # Shuffle the action list so that we select new
            # actions in case of ties. Then maximize
            random.shuffle(self.act_list)
            for a in self.act_list:
                temp_Q = Q[state][a]
                if temp_Q > best_Q:
                    best_Q = temp_Q
                    best_action = a
            return best_action
        
    # Function to choose function based on max value
    def action_callback(self, state):
        # If this is the first action callback of the epoch
        if not self.last_state:
            self.last_state = self.discreteState(state)
            # Choose A from S using policy derived from Q
            self.last_action = self.chooseAction(self.last_state, self.Q)
        else:
            action = self.last_action     

            # Choose A' from S' using policy derived from Q
            new_state  = self.discreteState(state)
            new_action = self.chooseAction(new_state, self.Q)

            # Q Learning
            self.Q[self.last_state][action] = self.Q[self.last_state][action] \
                                                   + self.alpha(self.N[self.last_state][action])*(self.last_reward + \
                                                                 self.gamma*self.Q[new_state][new_action] - \
                                                                 self.Q[self.last_state][action])
            self.N[self.last_state][action] += 1    
            # Update all vals
            self.last_action = new_action
            self.last_state  = new_state
            
        # Every time we perform an action, we've made an iteration
        self.numIters += 1
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        # Cumulative reward over the epoch
        # works better than individual reward
        if self.last_reward:
            self.last_reward += reward
        else:
            self.last_reward = reward

# Initial variables
# Note that epsilon is 1/iteration
# And alpha is 1/epoch minimized at .1 (we always wanna learn)
iters = 100
epsilon = lambda(x): 1./x
#epsilon = lambda(x): 1
alpha = lambda(x): min(.1, 1./x)

# Best discount factor I found, though not numerically verified
gamma = .95

# Best buckets I found, though not tested
buckets = 6

#restoredQ = np.load('Q2.pkl')
#restoredN = np.load('N2.pkl')
restoredQ = []
restoredN = []

learner = Learner(alpha, gamma, epsilon, 0, buckets, Q = restoredQ, N = restoredN)

scores = []

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

    # Just a very basic tracker of max score
    # Ideally for a good writeup, we should plot
    # Individual rewards and cumulative rewards over time
    # For different runs and maybe get things like avg score
    # Instead of max score. I was just pumped we were getting
    # So high to I decided to print the max out :)
    scores.append(swing.get_state()['score'])
    if swing.get_state()['score'] > learner.max_score:
        learner.max_score = swing.get_state()['score']

    # Reset the state of the learner.
    learner.reset()

# Print max score and enjoy :D
print learner.max_score
print float(np.sum(scores))/len(scores)
savedQ = np.array(learner.Q)
savedQ.dump('Q_bs.pkl')
savedN = np.array(learner.N)
savedN.dump('N_bs.pkl')
for i in range(len(learner.N)):
    if learner.N[i] != [1, 1]:
        print i, learner.N[i]


    
