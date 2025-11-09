# tic tac toe (strong)
import numpy as np
import random
import pickle

# === 対称性処理 ===
def canonical_state(state):
    """盤面の回転・反転を考慮して、最も小さい表現を返す"""
    board_2d = np.array(state).reshape(3,3)
    boards = []
    for k in range(4):
        rot = np.rot90(board_2d, k)
        boards.append(tuple(rot.flatten()))
        boards.append(tuple(np.fliplr(rot).flatten()))
    return min(boards)

# === 環境 ===
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

# === Qエージェント ===
class QAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return canonical_state(state)

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

# === 自己対戦学習 ===
env = TicTacToe()
player1 = QAgent(epsilon=0.2)
player2 = QAgent(epsilon=0.2)

episodes = 300000
for ep in range(episodes):
    state = env.reset()
    player = 1
    done = False
    while not done:
        current = player1 if player == 1 else player2
        actions = env.available_actions()
        action = current.choose_action(state, actions)
        next_state, reward, done = env.step(action, player)

        # 報酬処理
        if done:
            if env.winner == player:
                current.update(state, action, 1, next_state, True)
                # 負けた側にも報酬を与える（-1）
                other = player2 if player == 1 else player1
                other.update(state, action, -1, next_state, True)
            elif env.winner == -player:
                current.update(state, action, -1, next_state, True)
            else:
                current.update(state, action, 0, next_state, True)
        else:
            current.update(state, action, 0, next_state, False)

        state = next_state
        player *= -1

    # 進行表示
    if ep % 50000 == 0:
        print(f"学習中: {ep} / {episodes}")

# === 結合Qテーブル ===
# 両者のQを統合して最終AIに
final_Q = player1.Q.copy()
for k, v in player2.Q.items():
    if k not in final_Q:
        final_Q[k] = v
    else:
        final_Q[k] = (final_Q[k] + v) / 2

# 保存
with open("qtable_strong.pkl", "wb") as f:
    pickle.dump(final_Q, f)

print("✅ 強化学習完了！ Qテーブルを保存しました。")
