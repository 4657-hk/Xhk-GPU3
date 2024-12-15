# 马尔科夫决策过程 强化学习案例01
import numpy as np

# Define the grid world as a 4x4 matrix
grid_world = np.zeros((4, 4))

# Define the rewards for each state
punish = -5
finish_reward = 10

rewards = np.zeros((4, 4))
rewards[0, 0] = 0
rewards[0, 1] = punish
rewards[0, 2] = 0
rewards[0, 3] = finish_reward
rewards[1, 0] = 0
rewards[1, 1] = punish
rewards[1, 2] = 0
rewards[1, 3] = punish
rewards[2, 0] = 0
rewards[2, 1] = punish
rewards[2, 2] = 0
rewards[2, 3] = punish
rewards[3, 0] = 0
rewards[3, 1] = 0
rewards[3, 2] = 0
rewards[3, 3] = 0

# rewards = [[0, -5, 0, 10]
#            [0, -5, 0, -5]
#            [0, -5, 0, -5]
#            [0,  0, 0,  0]]

def choose_best_action(QxyList):
    best_action_list = np.argwhere(QxyList == np.max(QxyList))
    best_action = np.random.choice([unit[0] for unit in best_action_list])
    return best_action

# Define the discount factor
gamma = 0.9

# Define the value function for each state
V = np.zeros((4, 4))
V_new = np.zeros((4, 4))

# Define the action value function for each state-action pair
Q = np.zeros((4, 4, 4))

# Define the policy as a deterministic policy that always moves right
policy = np.zeros((4, 4, 4))
policy[:, :, 0] = 0
policy[:, :, 1] = 0
policy[:, :, 2] = 0
policy[:, :, 3] = 1

# Define the iteration parameters
num_iterations = 200
tolerance = 1e-6

# Perform value iteration to compute the optimal value function and policy
for i in range(num_iterations):
    # Compute the action value function for each state-action pair
    for x in range(4):
        for y in range(4):
            s_ = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            for a in range(4):
                x_, y_ = s_[a]
                if 0 <= x_ < 4 and 0 <= y_ < 4:
                    Q[x, y, a] = (rewards[x_, y_] + gamma * V[x_, y_])
                else:
                    Q[x, y, a] = 0
                Q[0, 3, a] = 0

    for a in range(4):
        print(f"action:{a}")
        print(f"Q[:, :, {a}]:\n{Q[:, :, a]}")

    V_new = np.max(Q, axis = 2)

    # Update the policy based on the new action value function
    new_policy = np.zeros((4, 4, 4))
    best_policy = np.zeros((4, 4))
    for x in range(4):
        for y in range(4):
            best_action = choose_best_action(Q[x, y, :])
            new_policy[x, y, best_action] = 1
            best_policy[x, y] = best_action

    print(f"round:{i}")
    print(f"best_policy:\n{best_policy}")

    if np.allclose(V, V_new):
        break

    print(f"V:\n{V_new}")
    print(f"-------------------"*3)

    V = V_new
    policy = new_policy

print(f"finish")