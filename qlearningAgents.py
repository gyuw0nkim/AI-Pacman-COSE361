# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        ## Q값을 저장할 qvalues 변수를 초기화
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        ## 주어진 상태에 대한 Q값 반환
        return(self.qvalues[(state, action)])

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        ## 가능한 모든 행동에 대한 Q값을 리스트로 저장해서, 그 중 최댓값을 반환
        QVal = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(QVal):
          return 0
        return max(QVal) ##리스트가 비어있지 않으면 최댓값을 찾아 반환
    

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalactions = self.getLegalActions(state)

        if not legalactions:
          return None

         # 주어진 상태에서 가능한 모든 action인 legalactions에 대해 모든 Q값을 구함
        QVal = [self.getQValue(state, action) for action in legalactions]
        maxQVal = max(QVal)

        # QVal == maxQVal인 action을 최적의 action으로 정의
        # 여러 opt action 중 random 함수를 사용해 무작위로 action 중 하나 return
        optactions = [action for action, QVal in zip(legalactions, QVal) if QVal == maxQVal]

        return random.choice(optactions)  
      

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # action 선택
        legalActions = self.getLegalActions(state)
        action = None

        # Epsilon-greedy action 선택
        # flipcoin이 true 반환 시 무작위 선택
        # false라면 현재 상태에서 최적인 것으로 생각되는 action return
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        else:
          return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q-learning 수식
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha * (R + gamma * max(Q(s',a')))
        self.qvalues[(state, action)] = (1-self.alpha)*self.getQValue(state, action) \
            + self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featureVectors = self.featExtractor.getFeatures(state, action)
        weightVectors = self.weights
        q_value = 0
        for key in featureVectors.keys():
            q_value += weightVectors[key]*featureVectors[key]
        return q_value

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        update = (reward + (self.discount*self.getValue(nextState))) - self.getQValue(state, action)
        featureVectors = self.featExtractor.getFeatures(state, action)

        for key in featureVectors.keys():
            self.weights[key] += self.alpha*update*featureVectors[key]


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
