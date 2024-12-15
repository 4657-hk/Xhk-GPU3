# Q-learning算法 强化学习案例02
import numpy as np

# Define the grid world
GRID_HEIGHT = 4
GRID_WIDTH = 4
START_STATE = (0, 0)
GOAL_STATE = (GRID_HEIGHT-1, GRID_WIDTH-1)
OBSTACLES = [(1, 1), (2, 2)]
ACTIONS = ['up', 'down', 'left', 'right']
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100

# Define the Q-learning parameters
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9

# Initialize the Q-table
q_table = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTIONS)))

# Define a function to choose an action based on the Q-values in the Q-table
def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON:
        # Choose a random action
        action = np.random.choice(ACTIONS)
    else:
        # Choose the action with the highest Q-value
        q_values = q_table[state[0], state[1], :]
        action = ACTIONS[np.argmax(q_values)]
    return action

# Define a function to update the Q-values in the Q-table based on the reward and next state
def update_q_table(state, action, reward, next_state):
    q_values = q_table[state[0], state[1], :]
    next_q_values = q_table[next_state[0], next_state[1], :]
    td_target = reward + GAMMA * np.max(next_q_values)
    td_error = td_target - q_values[ACTIONS.index(action)]
    q_values[ACTIONS.index(action)] += ALPHA * td_error
    print(f"==={state[0]}==={state[1]}==={ACTIONS.index(action)}==={ALPHA * td_error}===")
    q_table[state[0], state[1], :] = q_values

# Run the Q-learning algorithm
for episode in range(NUM_EPISODES):
    state = START_STATE
    done = False
    step = 0
    while not done and step < MAX_STEPS_PER_EPISODE:
        action = choose_action(state)
        if action == 'up':
            next_state = (max(state[0]-1, 0), state[1])
        elif action == 'down':
            next_state = (min(state[0]+1, GRID_HEIGHT-1), state[1])
        elif action == 'left':
            next_state = (state[0], max(state[1]-1, 0))
        elif action == 'right':
            next_state = (state[0], min(state[1]+1, GRID_WIDTH-1))
        if next_state in OBSTACLES:
            # Stay in the same state if the next state is an obstacle
            reward = -1
            next_state = state
        elif next_state == GOAL_STATE:
            # Receive a reward of +10 for reaching the goal state
            reward = 10
            done = True
        else:
            # Receive a reward of -1 for each step that doesn't lead to the goal state
            reward = -1
        update_q_table(state, action, reward, next_state)
        print(f"state:\n{state}")
        print(f"action:\n{action}")
        print(f"reward:\n{reward}")
        print(f"q_table0:\n{q_table[:,:,0]}")
        print(f"q_table1:\n{q_table[:, :, 1]}")
        print(f"q_table2:\n{q_table[:, :, 2]}")
        print(f"q_table3:\n{q_table[:, :, 3]}")
        state = next_state
        step += 1
