# tic tac toe (reinforcement learning)
import pickle
import numpy as np
import random

# --- 環境（同じ） ---
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.winner = None
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            self.done = True
            self.winner = -player
            return self.board.copy(), -10, True

        self.board[action] = player
        if self.check_winner(player):
            self.done = True
            self.winner = player
            return self.board.copy(), 1, True
        elif 0 not in self.board:
            self.done = True
            return self.board.copy(), 0, True
        return self.board.copy(), 0, False

    def check_winner(self, player):
        wins = [(0,1,2),(3,4,5),(6,7,8),
                (0,3,6),(1,4,7),(2,5,8),
                (0,4,8),(2,4,6)]
        return any(all(self.board[i]==player for i in w) for w in wins)

# --- Qエージェント ---
class QAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        qs = [self.Q.get((self.get_state_key(state), a), 0) for a in available_actions]
        max_q = max(qs)
        best_actions = [a for a, q in zip(available_actions, qs) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        key = (self.get_state_key(state), action)
        next_qs = [self.Q.get((self.get_state_key(next_state), a), 0)
                   for a in range(9)]
        max_next_q = max(next_qs) if not done else 0
        old_q = self.Q.get(key, 0)
        self.Q[key] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

# --- 学習 ---
env = TicTacToe()
agent = QAgent()
episodes = 100000

for ep in range(episodes):
    state = env.reset()
    player = 1
    done = False
    while not done:
        if player == 1:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            next_state, reward, done = env.step(action, player)
            agent.update(state, action, reward, next_state, done)
        else:
            actions = env.available_actions()
            action = random.choice(actions)
            next_state, reward, done = env.step(action, player)
        state = next_state
        player *= -1

print("Well done! Q table size:", len(agent.Q))

# 保存
with open("qtable.pkl", "wb") as f:
    pickle.dump(agent.Q, f)
print("Q table saved!")