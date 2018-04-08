# multiAgents.py
# --------------
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
# Project 2 pt 1 ReflexAgent & Minimax
# Project 2 pt 2 Alpha-Beta Pruning, Expectimax, & Better Evaluation Function
# Amelie Cameron
# CSC 665-01 Professor Yoon
# Due 04/08/18

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # set action score to zero
        total_score = 0.0
        # set distance to infinitely large value
        min_ghost_dist = float("inf")
        # loop through ghost states
        for ghostState in newGhostStates:
            # get position of ghosts
            ghost_pos = ghostState.getPosition()
            # calculate manhattan distance from player to ghost
            ghost_distance = util.manhattanDistance(newPos, ghost_pos)
            # set minimum distance if manhattan distance is less
            if ghost_distance < min_ghost_dist:
                min_ghost_dist = ghost_distance
            # add ghost distance to total score
            total_score += ghost_distance

        # if ghost is very close to pacman return distance
        if min_ghost_dist < 3:
            return min_ghost_dist

        # calculate the distance by getting food list and finding manhattan distance to each food dot
        food_distances = [util.manhattanDistance(newPos, food_pos) for food_pos in currentGameState.getFood().asList()]

        # if there is still food on the board, find the closest food
        if len(food_distances) > 0:
            # find closest food
            closest_food = min(food_distances)
            # return higher priority for close food rather than ghosts
            return 100.0 - closest_food
        return total_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # find how many agents
        self.game_agents = gameState.getNumAgents()

        # start with pacman agent at the root (depth 0)
        path = self.max_value(gameState, 0, 0)
        return path[1]

    def max_value(self, state, agent, depth):
        # each vertex has a score with an action
        max_score = float('-inf')
        max_action = "Stop"

        # loop through all actions for agent
        for action in state.getLegalActions(agent):

            # compute new depth
            new_depth = depth + 1

            # iterate through to next agent
            new_agent = new_depth % self.game_agents

            # get score for action
            cur_score = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth)

            # keep track of current
            # max score and update if current score is max
            if cur_score > max_score:
                max_score = cur_score
                max_action = action
        # return best score and action associated with it
        return max_score, max_action

    def min_value(self, state, agent, depth):
        # each vertex has a score with an action
        min_score = float('inf')
        min_action = "Stop"

        # loop through all actions for agent
        for action in state.getLegalActions(agent):

            # compute new depth
            new_depth = depth + 1

            # iterate though to next agent
            new_agent = new_depth % self.game_agents

            # get score for action
            cur_score = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth)

            # update if current score is minimum
            if cur_score < min_score:
                min_score = cur_score
                min_action = action
        # return min score and action associated with it
        return min_score, min_action

    def compute_score(self, state, agent, depth):
        # iterated through all depths and game agents
        max_possible = depth >= self.depth*self.game_agents
        # evaluate if game is over
        if max_possible or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        # if pacman, return max score
        if agent == 0:
            return self.max_value(state, agent, depth)[0]
        # if ghost, return min score
        else:
            return self.min_value(state, agent, depth)[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # find how many agents
        self.game_agents = gameState.getNumAgents()

        # start with pacman agent at the root (depth 0)
        # set alpha to negative infinity
        # set beta to positive infinity
        path = self.max_value(gameState, 0, 0, float('-inf'), float('inf'))
        return path[1]

    def max_value(self, state, agent, depth, alpha, beta):
        # assign value to negative infinity and value action to stop
        v = float('-inf')
        v_action = 'Stop'

        # loop through successor states
        for action in state.getLegalActions(agent):

            # compute new depth
            new_depth = depth + 1

            # iterate through to next agent
            new_agent = new_depth % self.game_agents

            # get current score for action at given depth of given agent and given alpha and beta values
            cur_v = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth, alpha, beta)

            # find max, if current value is bigger than set value, reset v
            if cur_v > v:
                v = cur_v
                v_action = action

            # if current score is maximum, return
            if v > beta:
                return v, v_action
            # set new alpha value to the larger of either alpha or v
            alpha = max(alpha, v)
        # return maximum value
        return v, v_action

    def min_value(self, state, agent, depth, alpha, beta):
        # each vertex has a score with an action
        v = float('inf')
        v_action = "Stop"

        # loop through all actions for agent
        for action in state.getLegalActions(agent):
            # compute new depth
            new_depth = depth + 1

            # iterate though to next agent
            new_agent = new_depth % self.game_agents

            # get current score for action at given depth of given agent and given alpha and beta values
            cur_v = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth, alpha, beta)

            # find min, if current value is smaller than set value, reset v
            if cur_v < v:
                v = cur_v
                v_action = action

            # if current score is minimum, return
            if v < alpha:
                return v, v_action
            # set new beta value to the smaller of either beta or v
            beta = min(beta, v)
        # return minimum value
        return v, v_action

    def compute_score(self, state, agent, depth, alpha, beta):
        # iterated through all depths and game agents
        max_possible = depth >= self.depth * self.game_agents
        # evaluate if game is over
        if max_possible or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        # if pacman, return max value
        if agent == 0:
            return self.max_value(state, agent, depth, alpha, beta)[0]
        # if ghosts, return min value
        else:
            return self.min_value(state, agent, depth, alpha, beta)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # find how many agents
        self.game_agents = gameState.getNumAgents()

        # start with pacman agent at the root (depth 0)
        path = self.max_value(gameState, 0, 0)
        return path[1]

    def max_value(self, state, agent, depth):
        # each vertex has a score with an action
        max_score = float('-inf')
        max_action = "Stop"

        # loop through all actions for agent
        for action in state.getLegalActions(agent):

            # compute new depth
            new_depth = depth + 1

            # iterate through to next agent
            new_agent = new_depth % self.game_agents

            # get score for action given agent and depth
            cur_score = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth)

            # update if current score is max
            if cur_score > max_score:
                max_score = cur_score
                max_action = action
        # return max value
        return max_score, max_action

    def rand_value(self, state, agent, depth):
        # find legal actions for state
        legal_actions = state.getLegalActions(agent)
        # create list of possible scores for states
        score_list = {}
        cur_score = 0
        # loop through possible actions
        for action in legal_actions:
            # increment depth to check all levels
            new_depth = depth + 1
            # iterate through game agents
            new_agent = new_depth % self.game_agents
            # assign scores for each action to list given agent and depth
            score_list[action] = self.compute_score(state.generateSuccessor(agent, action), new_agent, new_depth)
        # loop through list and sum the scores
        for s in score_list.values():
            cur_score += s
        # find the average score given the sum and and the number of elements in the list
        avg_score = float(cur_score) / float(len(score_list))
        # return the float value average for ghost score
        return avg_score, None

    def compute_score(self, state, agent, depth):
        # iterated through all depths and game agents
        max_possible = depth >= self.depth * self.game_agents
        # evaluate is game is over
        if max_possible or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        # if pacman, return optimal value
        if agent == 0:
            return self.max_value(state, agent, depth)[0]
        # if ghosts, return random suboptimal value
        else:
            return self.rand_value(state, agent, depth)[0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # current pacman position
    curPos = currentGameState.getPacmanPosition()
    # current ghost states
    ghostStates = currentGameState.getGhostStates()

    # set action score to zero
    total_score = 0.0
    # make a list of closest ghosts to pacman
    closest_ghosts = []
    # loop through ghost states
    for ghostState in ghostStates:
        # get position of ghosts
        ghost_pos = ghostState.getPosition()
        # calculate manhattan distance from player to ghost
        ghost_distance = util.manhattanDistance(curPos, ghost_pos)
        # if ghost is next to pacman, add it to closest ghosts list
        if ghost_distance == 0:
            closest_ghosts.append(0)
        # else rank each ghost based on distance
        else:
            # if distance from pacman is further, the result is higher
            # use this ghost score to discount food score
            # if the ghost is close by then the pacman should avoid ghost
            # rather than eating food
            g_score = -1/ghost_distance
            closest_ghosts.append(g_score)
    # prioritize closest ghost
    ghost_score = min(closest_ghosts)
    # if food is further away it's going to be a lower score
    # find manhattan distance between pacman and food in list
    # multiply values by -1 so that the max will return the closest food to the pacman
    food_distances = [-1*util.manhattanDistance(curPos, food_pos) for food_pos in currentGameState.getFood().asList()]
    # if there is food still on the board, food score will return the closest food
    if len(food_distances):
        food_score = max(food_distances)
    # if food is gone, food score is high
    else:
        food_score = 1000
    # get current score given current game state
    current_score = currentGameState.getScore()
    # get linear combination of food proximity, ghost proximity, and possible score in order to chose optimal move
    return food_score + ghost_score + current_score

# Abbreviation
better = betterEvaluationFunction

