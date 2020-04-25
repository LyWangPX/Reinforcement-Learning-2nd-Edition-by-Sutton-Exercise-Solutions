# Example 6.5 of the book "Reinforcement Learning: an Introduction"
# Maxime Xuereb - Python implementation
import numpy as np

class Windy_Gridworld:

    def __init__(self):
        self.world = np.zeros((7, 10))  # World
        # The world contains in each cell the power of the wind
        # Positive means it will go up in the real world, i.e. down in the i axis and still in the j axis
        for i in range(len(self.world)):
            self.world[i][3:6] = 1
            self.world[i][6:8] = 2
            self.world[i][8:9] = 2
        self.start_cell = [3, 0]  # Cells where we start
        self.end_cell = [3, 7]  # Cell where we end
        self.gamma = 1  # Gamma
        # Initialize Q matrix: actions are in this order: up - right - down - left (4 actions)
        self.Q = np.zeros((7, 10, 4))
        # Initialize policy (will be epsilon-soft)
        self.epsilon = 0.1
        self.policy = 0.25 * np.ones((7, 10, 4))
        # Initialize step size in (0,1]
        self.alpha = 0.2
        # Initialize number of episodes
        self.episodes = 500

        # Start the Sarsa on-policy TD
        self.sarsa()

    def sarsa(self):
        # Loop for each episode
        for ep in range(self.episodes):
            print("START OF EPISODE " + str(ep + 1))
            # The initial state is fixed, choose initial action based on the epsilon-soft policy
            current_state = self.start_cell
            current_action = int(np.random.choice(np.arange(4), 1,
                                                  p=self.policy[self.start_cell[0]][self.start_cell[1]]))
            terminal_state = False
            number_states = 0
            while not terminal_state:
                number_states = number_states + 1
                # Take action A, observe R,S'
                new_state = [-1, -1]
                move_to_make = self.convert_to_move(current_action)
                after_move_i = current_state[0]+ move_to_make[0]
                # If we go off the grid, I simply put us back in the grid, as if the grid contained walls that
                # cannot be crossed.
                if after_move_i < 0:
                    after_move_i = 0
                elif after_move_i >= 7:
                    after_move_i = 6
                new_state[0] = current_state[0] + move_to_make[0] - self.world[after_move_i][current_state[1]]
                new_state[1] = current_state[1] + move_to_make[1]
                if new_state[0] < 0:
                    new_state[0] = 0
                elif new_state[0] >= 7:
                    new_state[0] = 6
                if new_state[1] < 0:
                    new_state[1] = 0
                elif new_state[1] >= 10:
                    new_state[1] = 9

                if new_state[0] - self.end_cell[0] == 0 and new_state[1] - self.end_cell[1] == 0:
                    current_reward = 1
                    terminal_state = True
                else:
                    current_reward = -1
                new_state[0] = int(new_state[0])
                new_state[1] = int(new_state[1])
                # Choose A' from S' and the policy
                new_action = int(np.random.choice(np.arange(4), 1,p=self.policy[new_state[0]][new_state[1]]))
                # Update the Q value
                self.Q[current_state[0]][current_state[1]][current_action] = self.Q[current_state[0]][current_state[1]][current_action] \
                                                                             + self.alpha*(current_reward
                                                                                           + self.gamma*self.Q[new_state[0]][new_state[1]][new_action]                                                                              - self.Q[current_state[0]][current_state[1]][current_action])
                # Update the policy
                best_action = 0
                best_value = self.Q[current_state[0]][current_state[1]][0]
                for a in range(1,4):
                    current_value = self.Q[current_state[0]][current_state[1]][a]
                    if current_value > best_value:
                        best_action = a
                        best_value = current_value
                for a in range(4):
                    if a == best_action:
                        self.policy[current_state[0]][current_state[1]][a] = 1 - self.epsilon + (self.epsilon/4)
                    else:
                        self.policy[current_state[0]][current_state[1]][a] = self.epsilon/4
                # S <-- S' and A <-- A'
                current_state = new_state
                current_action = new_action
            print("Number of states: " + str(number_states))

    def convert_to_move(self, action_number):
        if action_number == 0:  # Up
            return [-1, 0]
        elif action_number == 1:  # Right
            return [0, 1]
        elif action_number == 2:  # Down
            return [1, 0]
        elif action_number == 3:  # Left
            return [0, -1]
        else:  # Problem
            print("Problem in the action taken - value not supposed to exist")
            return None

problem = Windy_Gridworld()