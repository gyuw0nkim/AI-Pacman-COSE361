# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
            counter = util.Counter()
            for state in self.mdp.getStates():
                #terminal state에서의 value는 0이다.
                if self.mdp.isTerminal(state):
                    counter[state] = 0
                else:
                    maxVal = -float('inf') ##maxVal을 음의 무한대로 초기화
                    # V[k][s] = [E(모든 s'에 대한 합) T(s,a,s')[R(s,a,s') + gama*V[k-1][s']]] 의 max값
                    for action in self.mdp.getPossibleActions(state):
                        T = self.mdp.getTransitionStatesAndProbs(state, action)
                        QVal = 0
                        for sp in T:
                            QVal += sp[1] * (self.mdp.getReward(state, action, sp[1]) + self.discount * self.values[sp[0]])
                        maxVal = max(QVal, maxVal)

                    if maxVal != -float('inf'):
                        counter[state] = maxVal

            self.values = counter   



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        QVal = 0
        # Q(s,a) = E(s'에 대한 모든 action의 합) T(s,a,s')[R(s,a,s') + gama*V(s')]
        for StateProb in self.mdp.getTransitionStatesAndProbs(state, action):
            QVal += StateProb[1] * (self.mdp.getReward(state, action, StateProb[0]) + self.discount * self.values[StateProb[0]])
            
        return QVal
    
        #util.raiseNotDefined()
    
    
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        if self.mdp.isTerminal(state):
            return None
        
        optAction = self.mdp.getPossibleActions(state)[0]
        totalactions = {}

        for action in self.mdp.getPossibleActions(state):
            totalactions[action] = self.computeQValueFromValues(state, action)

        return max(totalactions, key = totalactions.get)
        
        #util.raiseNotDefined()
 

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
        