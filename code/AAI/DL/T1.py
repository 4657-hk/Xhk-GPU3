## 1. 程序初始化
import time
import random
import pickle
import logging
import argparse
import itertools
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
from gym import spaces
from gym.utils import seeding

## 2. 预置训练参数初始化
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)  # 学习率
parser.add_argument("--gamma", type=float, default=0.99)           # 经验折扣率
parser.add_argument("--epochs", type=int, default=10000)           # 迭代多少局数
parser.add_argument("--buffer_size", type=int, default=200000)     # replaybuffer大小
parser.add_argument("--batch_size", type=int, default=256)         # batchsize大小
parser.add_argument("--pre_train_model", type=str, default=None)   # 是否加载预训练模型
parser.add_argument("--use_nature_dqn", type=bool, default=True)   # 是否采用nature dqn
parser.add_argument("--target_update_freq", type=int, default=250) # 如果采用nature dqn，target模型更新频率
parser.add_argument("--epsilon", type=float, default=0.9)          # 探索epsilon取值
args, _ = parser.parse_known_args()
#！！！！注意参数是否实际使用，先读代码理解训练过程！！！！

## 3. 创建环境
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass


def stack(flat, layers=16):
    """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
    # representation is what each layer represents
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # layered is the flat board repeated layers times
    layered = np.repeat(flat[:, :, np.newaxis], layers, axis=-1)

    # Now set the values in the board to 1 or zero depending whether they match representation.
    # Representation is broadcast across a number of axes
    layered = np.where(layered == representation, 1, 0)

    return layered

def unstack(layered):
    """Convert an [4, 4, layers] representation back to [4, 4]."""
    # representation is what each layer represents
    layers = layered.shape[-1]
    representation = 2 ** (np.arange(layers, dtype=int) + 1)

    # Use the representation to convert binary layers back to original values
    original = np.zeros((4, 4), dtype=int)
    for i in range(layers):
        # Convert the result to integer before adding
        addition = (layered[:, :, i] * representation[i]).astype(int)
        original += addition

    return original


# 游戏环境
class Game2048Env(gym.Env):
    metadata = {'render.modes': ['ansi', 'human', 'rgb_array']}

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (self.w, self.h, layers), int)
        self.set_illegal_move_reward(-20)
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2 ** self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            if score > 0:
                score = score
                # score = math.log2(score)
            if score < 0:
                score = 0
            self.score += score
            assert score <= 2 ** (self.w * self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)

        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            reward = self.illegal_move_reward
            # reward=0

        # print("Am I done? {}".format(done))
        info['highest'] = self.highest()

        # Return observation (board state), reward, done and info dict
        return stack(self.Matrix), reward, done, info
    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix)




    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def board_total(self):
        """Calculate the total value of all tiles on the board."""
        return np.sum(self.Matrix)


    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the maximum score that [would have] got from the move."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        scores = []  # 修改为列表，用于存储每次移动得到的分数
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two  # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                scores.append(ms)  # 添加到分数列表中
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                scores.append(ms)  # 添加到分数列表中
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if not changed:
            raise IllegalMove
            # 打印分数列表和最大分数

        # 获取列表中的最大值作为 move_scores 返回
        move_scores = max(scores) if scores else 0
        # print("Scores from this move:", scores)
        # print("Maximum score from this move:", move_scores)
        return move_scores

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        # if self.max_tile is not None and self.highest() == self.max_tile:
        #     return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board

## 定义DQN算法
# 定义用于低奖励情况的神经网络
class NetLowReward(nn.Module):
    def __init__(self, obs, available_actions_count):
        super(NetLowReward, self).__init__()
        self.conv1 = nn.Conv2d(obs, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(16, available_actions_count)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc1(x.view(x.shape[0], -1))
        return x

# 定义用于高奖励情况的神经网络
class NetHighReward(nn.Module):
    def __init__(self, obs, available_actions_count):
        super(NetHighReward, self).__init__()
        self.conv1 = nn.Conv2d(obs, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(16, available_actions_count)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc1(x.view(x.shape[0], -1))
        return x


class DQN:
    def __init__(self, args, obs_dim, action_dim, net, memory):
        self.args = args
        self.behaviour_model = net(obs_dim, action_dim).to(device)
        # self.behaviour_model = torch.load("512-rollout-model.pt").to(device)  # 加载并使用已训练的模型
        self.target_model = net(obs_dim, action_dim).to(device)
        # self.target_model =  torch.load("512-rollout-model.pt").to(device)
        self.optimizer = torch.optim.Adam(self.behaviour_model.parameters(), args.learning_rate)
        self.criterion = nn.MSELoss()
        self.action_dim = action_dim
        self.learn_step_counter = 0
        self.memory = memory
        self.post_full_learn_counter = 0  # 初始化达到最大容量后的学习计数器
        self.states_for_cosine_similarity = []  # 存储用于余弦相似度计算的状态
        self.q_values_for_cosine_similarity = []  # 存储这些状态的旧 Q-
        self.new_q_values_for_cosine_similarity=[]  # 更新列表2中的Q值
        self.global_learn_counter = 0
        self.save_graph_counter = 0  # 新的计数器


    def learn(self):
        # 当经验池未满时，按原有逻辑进行学习
        if self.memory.size < self.memory.buffer_size:
            if self.memory.size <= 5000 and self.memory.size >= self.args.batch_size:
                # 从经验池中随机抽取batch_size个样本进行学习
                s1, a, s2, done, r = self.memory.get_sample(self.args.batch_size)
                self.update_model(s1, a, s2, done, r)
            elif self.memory.size > 5000 and self.memory.size % 5000 == 0:
                for _ in range(80):  # 进行80次学习
                    s1, a, s2, done, r = self.memory.get_sample(self.args.batch_size)
                    self.update_model(s1, a, s2, done, r)

        # 当经验池达到或超过最大容量时，每隔5000步进行80次学习
        else:
            self.post_full_learn_counter += 1  # 更新计数器
            if self.post_full_learn_counter % 5000 == 0:
                for _ in range(80):  # 80
                    s1, a, s2, done, r = self.memory.get_sample(self.args.batch_size)
                    self.update_model(s1, a, s2, done, r)

        # 第一次经验池大小达到4999时，存储100个状态及其Q值
        if self.memory.size == 4999:
            sampled_indices = np.random.choice(range(self.memory.size), 100, replace=False)
            states = torch.FloatTensor(self.memory.s1[sampled_indices]).to(device)
            with torch.no_grad():
                q_values = self.behaviour_model(states)
            self.states_for_cosine_similarity = states
            self.q_values_for_cosine_similarity = q_values
            # print('这是旧的',q_values)

        # 当经验池大于4999但未满，或者经验池已满时，每隔5000步进行更新q
        if self.memory.size > 4999:
            # 更新全局计数器
            self.global_learn_counter += 1
            if self.global_learn_counter % 5000 == 0:
                # 重新计算Q值
                with torch.no_grad():
                    new_q_values = self.behaviour_model(self.states_for_cosine_similarity)
                # 更新列表2中的Q值
                self.new_q_values_for_cosine_similarity = new_q_values
                print(new_q_values)
                # # 计算并存储余弦相似度
                # cosine_similarity = self.calculate_cosine_similarity()


    def update_model(self, s1, a, s2, done, r):
        s1 = torch.FloatTensor(s1).to(device)
        s2 = torch.FloatTensor(s2).to(device)
        r = torch.FloatTensor(r).to(device)
        a = torch.LongTensor(a).to(device)
        done = torch.FloatTensor(done).to(device)

        if self.learn_step_counter % self.args.target_update_freq == 0:
            self.target_model.load_state_dict(self.behaviour_model.state_dict())
        self.learn_step_counter += 1

        if self.args.use_nature_dqn:
            q = self.target_model(s2).detach()
        else:
            q = self.behaviour_model(s2)

        target_q = r + (self.args.gamma * (1 - done)).to(device) * q.max(1)[0]
        eval_q = self.behaviour_model(s1).gather(1, a.view(-1, 1))
        target_q = target_q.view(-1, 1)
        loss = self.criterion(eval_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_with_new_q(self, state, new_q):
        # 将状态转换为张量格式
        state_tensor = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        # 计算网络当前的 Q 值
        current_q_values = self.behaviour_model(state_tensor)
        # 将新的 Q 值转换为张量格式
        new_q_values = torch.FloatTensor(new_q).to(device).view_as(current_q_values)
        # 计算损失并更新网络
        loss = self.criterion(current_q_values, new_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_cosine_similarity(self):
        # 使用更新后的模型计算新的 Q-值
        with torch.no_grad():
            new_q_values = self.behaviour_model(self.states_for_cosine_similarity)

        # 计算余弦相似度
        cosine_similarities = []
        for old_q, new_q in zip(self.q_values_for_cosine_similarity, new_q_values):
            cosine_similarity = 1 - cosine(old_q.cpu().numpy(), new_q.cpu().numpy())
            cosine_similarities.append(cosine_similarity)

        average_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
        return average_cosine_similarity

    def calculate_normalized_distance(self):
        # 计算新旧Q值之间的归一化欧几里得距离
        normalized_distances = []
        with torch.no_grad():
            new_q_values = self.behaviour_model(self.states_for_cosine_similarity)

        for old_q, new_q in zip(self.q_values_for_cosine_similarity, new_q_values):
            # print('这是旧',old_q)
            # print('这是新',new_q)
            distance = torch.norm(new_q - old_q).item() / torch.norm(old_q).item()
            normalized_distances.append(distance)

        average_distance = sum(normalized_distances) / len(normalized_distances)
        return average_distance

    def get_action(self, state, explore=True):
        if explore:
            # 使用模型预测动作的概率分布
            logits = self.behaviour_model(torch.FloatTensor(state).to(device))
            policy = F.softmax(logits, dim=1)
            m = Categorical(policy)
            # 随机选择一个动作，基于模型的概率分布
            action = m.sample().item()
        else:
            # 如果不探索，直接选择最佳动作
            with torch.no_grad():
                q = self.behaviour_model(torch.FloatTensor(state).to(device))
                _, action = torch.max(q, 1)
                action = action.item()

        return action

    def test_directions(self, special_buffer, model, num_trials, target_score=512):
        all_states = special_buffer.get_all_states_from_b_to_pos()
        # 去重处理
        unique_states = []
        for state in all_states:
            if not any(np.array_equal(state, unique_state) for unique_state in unique_states):
                unique_states.append(state)

        # print("去重后的状态数量:", len(unique_states))

        direction_results = {"up": [], "down": [], "left": [], "right": []}
        found_target = False
        key_state = None  # 用于存储关键状态
        key_direction = None  # 新增变量记录关键方向

        # 反向遍历所有去重后的状态
        for state in reversed(unique_states):
            unstacked_state = unstack(state)
            # print('检查的状态:', unstacked_state)
            s=unstacked_state.copy()

            temp_results = {"up": None, "down": None, "left": None, "right": None}

            # 检查每个方向
            for direction in temp_results:
                # print('检查方向:', direction)
                avg_rounds, reached_target, target_direction = self.run_trials(unstacked_state, model, direction, num_trials, target_score)

                temp_results[direction] = avg_rounds

                if reached_target and not found_target:
                    found_target = True
                    key_state = s
                    key_direction = target_direction


            if found_target:
                # 如果找到关键节点，将临时结果复制到最终结果中
                for direction in direction_results:
                    direction_results[direction] = [temp_results[direction]]
                break

        if not found_target:
            # 如果没有找到目标分数，将所有方向的分数设置为0
            for direction in direction_results:
                direction_results[direction] = [0]

        return direction_results, key_state, key_direction

    def run_trials(self, initial_state, model, direction, num_trials, target_score):
        total_rounds = 0
        reached_target = False

        for _ in range(num_trials):
            # 重置环境到初始状态
            env.Matrix = initial_state
            # print('初始状态:', initial_state)
            env.score = 0
            s = stack(initial_state)
            rounds = 0
            done = False

            # 执行第一步
            s, _, done, _, _, _ = env.step(self.direction_to_action(direction))
            last_s = None

            while not done:
                s_tensor = torch.FloatTensor(np.expand_dims(s, axis=0)).to(device)
                with torch.no_grad():
                    q = model(s_tensor)
                    a = torch.argmax(q, 1).item()
                s, _, done, _, _, mm = env.step(a)
                if env.highest() > target_score:
                    reached_target = True

                if np.array_equal(s, last_s):
                    break
                last_s = s.copy()
                rounds += 1

            total_rounds += rounds

        return total_rounds / num_trials, reached_target,direction

    def direction_to_action(self, direction):
        action = {"up": 0, "right": 1,"down": 2, "left": 3}[direction]
        # print(f"Direction: {direction}, Action: {action}")
        return action


class ReplayBuffer:
    def __init__(self, buffer_size, obs_space):
        self.s1 = np.zeros(obs_space, dtype=np.float32)
        self.s2 = np.zeros(obs_space, dtype=np.float32)
        self.a = np.zeros(buffer_size, dtype=np.int32)
        self.r = np.zeros(buffer_size, dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)

        # replaybuffer大小
        self.buffer_size = buffer_size
        self.size = 0
        self.pos = 0

    # 不断将数据存储入buffer
    def add_transition(self, s1, action, s2, done, reward):
        self.s1[self.pos] = s1
        self.a[self.pos] = action
        if not done:
            self.s2[self.pos] = s2
        self.done[self.pos] = done
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    # 随机采样一个batchsize
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.done[i], self.r[i]

    # 清空数据
    def clear(self):
        self.s1.fill(0)
        self.s2.fill(0)
        self.a.fill(0)
        self.r.fill(0)
        self.done.fill(0)
        self.size = 0
        self.pos = 0


class SpecialReplayBuffer:
    def __init__(self, buffer_size, obs_space):
        self.s1 = np.zeros(obs_space, dtype=np.float32)
        self.pos = 0
        self.buffer_size = buffer_size
        self.size = 0
        self.pointer_b_updated = False  # 用于标记是否已更新

    def add_state(self, s1):
        self.s1[self.pos] = s1
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        if not self.pointer_b_updated:
            self.pointer_b = self.pos
            self.pointer_b_updated = True  # 标记已更新

    def reset_pointer_b(self):
        self.pointer_b_updated = False  # 重置标记

    def get_last_states(self, n):
        print("当前指针位置:", self.pos)
        indices = [(self.pos - i - 1) % self.buffer_size for i in range(n)]
        return self.s1[indices]

    def get_first_states(self, n):
        print("当前指针位置:", self.pointer_b)
        if self.size < n:
            return self.s1[:self.size]  # 如果存储的状态少于n个，返回所有状态
        start_index = self.pointer_b
        indices = [(start_index + i) % self.buffer_size for i in range(n)]  # 计算需要返回的状态的索引
        return self.s1[indices]  # 返回这些状态


    def get_all_states_from_b_to_pos(self):
        if self.pointer_b <= self.pos:
            # 如果头指针在尾指针之前或相等，则直接返回这个范围内的状态
            return self.s1[self.pointer_b:self.pos]
        else:
            # 如果头指针在尾指针之后，则需要分两部分返回
            return np.concatenate((self.s1[self.pointer_b:], self.s1[:self.pos]), axis=0)

# 测试模型函数:
def infer_model(model, env, threshold=1024, num_trials=100):
    random_action_count = 0
    model_action_count = 0
    success_count = 0

    for _ in range(num_trials):
        s = env.reset()
        last_s = None
        actions_taken = []  # 初始化已尝试的动作列表

        while True:
            s_tensor = torch.FloatTensor(np.expand_dims(s, axis=0)).to(device)
            if np.array_equal(last_s, s):
                # 去除已尝试的动作
                available_actions = [i for i in range(4) if i not in actions_taken]

                # 如果没有剩余动作尝试，结束当前尝试
                if not available_actions:
                    break

                # 选择一个新的动作
                a = random.choice(available_actions)
                actions_taken.append(a)  # 记录尝试过的动作
                random_action_count += 1
            else:
                with torch.no_grad():
                    logits = model(s_tensor)
                    policy = F.softmax(logits, dim=1)
                    m = Categorical(policy)
                    a = m.sample().item()
                model_action_count += 1
                # 重置动作列表
                actions_taken = []

            last_s = np.array(s)
            s_, r, done, info = env.step(a)
            if done:
                if env.highest() >= threshold:
                    success_count += 1
                break
            s = s_
    return success_count, random_action_count, model_action_count

# 胜率绘制函数
def plot_success_rate(success_rate_iterations, success_rates, filename):
    # 绘制 Success Rate
    plt.plot(success_rate_iterations, success_rates, marker='o', label='Success Rate')
    for i, txt in enumerate(success_rates):
        plt.text(success_rate_iterations[i], success_rates[i], f"{txt:.3f}")

    # 设置图表标题和坐标轴标签
    plt.xlabel("Iteration")
    plt.ylabel("Max Reward ")
    plt.title("Max Reward, c Value. Iteration")

    # 显示图例和网格
    plt.legend()
    plt.grid()

    # 保存图表
    plt.savefig(filename)
    plt.close()

## 5. 训练模型

# 初始化环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = Game2048Env()
# 初始化replay buffer
replay_buffer_low = ReplayBuffer(buffer_size=args.buffer_size, obs_space=(args.buffer_size, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]))
# replay_buffer_high = ReplayBuffer(buffer_size=args.buffer_size, obs_space=(args.buffer_size, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]))
# filter_pool= ReplayBuffer(buffer_size=10000, obs_space=(args.buffer_size, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2]))


# 为低奖励和高奖励情况分别初始化 DQN 实例
DQN_low = DQN(args, obs_dim=env.observation_space.shape[2], action_dim=env.action_space.n, net=NetLowReward, memory=replay_buffer_low)
# DQN_high = DQN(args, obs_dim=env.observation_space.shape[2], action_dim=env.action_space.n, net=NetHighReward, memory=replay_buffer_high)

print('\ntraining...')

begin_t = time.time()
max_reward = 0
max_rewards = []  # 存储每次拐点的 max_reward 值

max_reward_iterations = []  # 存储每次拐点的迭代次数
current_interval_rewards = []

best_s, best_r, best_done, best_info = None, None, None, None
best_model = None
standard_scores = []
success_rates = []  # 存储每次拐点的 success_rate 值
success_rate_iterations = []  # 存储每次拐点的迭代次数
success_rates_256 = []
success_rates_512 = []


mark=0
best_success_rate = 0
for i_episode in range(400000):
    # 每局开始，重置环境
    s = env.reset()
    # 累计奖励值
    ep_r = 0
    c = 0
    # 初始化阶段标准分数为2
    standard_score = 2
    # 默认使用DQN_low模型
    current_DQN = DQN_low
    m_history = []  # 初始化记录每一步的m值的列表
    intervention_count = 0  # 初始化人工干预计数器
    while True:
        # 计算动作
        a = current_DQN.get_action(np.expand_dims(s, axis=0))
        # 执行动作
        s_, r, done, info = env.step(a)
        #unstack(s)
        replay_buffer_low.add_transition(s, a, s_, done, r)
        current_DQN.learn()

        max_tile = env.highest()

        # 如果得分大于等于标准分数的2倍，更新标准分数
        if max_tile >= 2 * standard_score:
            standard_score = max_tile

        if done:
            print('Ep: ', i_episode, '| Standard_score: ', standard_score, 'R:', intervention_count)
            standard_scores.append(max_tile)

            if i_episode % 5000 == 0:
                success_count, _, _ = infer_model(current_DQN.behaviour_model, env)
                success_rate = success_count / 100
                success_rates.append(success_rate)
                success_rate_iterations.append(i_episode)

                success_count_256, _, _ = infer_model(current_DQN.behaviour_model, env,256,100)
                success_rate_256=success_count_256/100
                success_rates_256.append(success_count_256)

                success_count_512, _, _ = infer_model(current_DQN.behaviour_model, env, 512, 100)
                success_rate_512 = success_count_512 / 100
                success_rates_512.append(success_count_512)

                if success_rate >= best_success_rate:
                    best_success_rate = success_rate
                    print(f"best_success_rate: {best_success_rate}")
                    best_model = current_DQN.behaviour_model

                    model_data = {
                        'model': best_model,
                        'replay_buffer': replay_buffer_low,
                        'other_data': {
                            'standard_score': standard_score,
                            'success_rate': success_rate,
                            'rounds': c
                        }
                    }

                    with open("Test4.pkl", 'wb') as file:
                        pickle.dump(model_data, file)
            break
        s = s_
