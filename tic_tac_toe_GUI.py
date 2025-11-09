# tic tac toe (GUI ver)
# Let's play tic tac toe with AI
import tkinter as tk
import numpy as np
import random
import pickle


# ====== 簡易 Qエージェント（学習済み想定 or ランダム） ======
class QAgent:
    def __init__(self, Q=None):
        self.Q = Q if Q else {}

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available_actions, epsilon=0.0):
        # 学習済みQがあれば最良手、なければランダム
        qs = [self.Q.get((self.get_state_key(state), a), 0) for a in available_actions]
        max_q = max(qs)
        best_actions = [a for a, q in zip(available_actions, qs) if q == max_q]
        return random.choice(best_actions)

# ====== 環境 ======
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0:空, 1:人間, -1:AI
        self.done = False
        self.winner = None
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action, player):
        if self.board[action] != 0:
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

# ====== GUI ======
class TicTacToeGUI:
    def __init__(self, root, agent):
        self.root = root
        self.agent = agent
        self.env = TicTacToe()

        self.root.title("TicTacToe vs AI")
        self.buttons = []
        self.create_board()

        self.info = tk.Label(root, text="Your turn (○)", font=("Arial", 14))
        self.info.grid(row=3, column=0, columnspan=3)

    def create_board(self):
        for i in range(9):
            btn = tk.Button(self.root, text=" ", font=("Arial", 32),
                            width=3, height=1,
                            command=lambda i=i: self.click_cell(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)

    def click_cell(self, idx):
        if self.env.done:
            return

        # 人間の手
        if self.env.board[idx] == 0:
            self.env.step(idx, 1)
            self.update_board()
            if self.env.done:
                self.show_result()
                return

            # AIの手
            self.root.after(500, self.ai_move)

    def ai_move(self):
        actions = self.env.available_actions()
        if not actions:
            return
        action = self.agent.choose_action(self.env.board, actions)
        self.env.step(action, -1)
        self.update_board()
        if self.env.done:
            self.show_result()

    def update_board(self):
        for i, v in enumerate(self.env.board):
            if v == 1:
                self.buttons[i].config(text="○", state="disabled", disabledforeground="blue")
            elif v == -1:
                self.buttons[i].config(text="x", state="disabled", disabledforeground="red")
            else:
                self.buttons[i].config(text=" ", state="normal")

    def show_result(self):
        if self.env.winner == 1:
            self.info.config(text="You win! 🎉")
        elif self.env.winner == -1:
            self.info.config(text="AI wins 😈")
        else:
            self.info.config(text="draw 🤝")

        for b in self.buttons:
            b.config(state="disabled")

# ====== 実行 ======
if __name__ == "__main__":
    root = tk.Tk()

    # ここで学習済みQを読み込み可能 (例: pickle.load(open("qtable.pkl","rb")))
    agent = QAgent()
    agent.Q = pickle.load(open("qtable_strong.pkl", "rb"))
    agent.epsilon = 0

    gui = TicTacToeGUI(root, agent)
    root.mainloop()
