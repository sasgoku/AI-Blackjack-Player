# 🎮 AI Blackjack Player

An AI-powered Blackjack player built using **Python and Reinforcement Learning (Q-Learning)**. The agent learns how to play Blackjack by interacting with a simulated game environment and improving its decisions through trial and error.

## 🚀 Features

- Blackjack game simulation.
- AI agent powered by Q-Learning.
- Learns when to **Hit** or **Stand**.
- Uses an epsilon-greedy strategy to balance exploration and exploitation.
- Trains over thousands of game episodes.
- Evaluates the agent's performance.
- Saves and loads the trained Q-table.
- Displays the learned strategy.
- Includes automated unit tests.

## 🧠 How It Works

The AI uses **Q-Learning**, a reinforcement learning algorithm.

1. The agent observes the current game state.
2. It chooses an action: Hit or Stand.
3. The game provides a reward based on the result.
4. The agent updates its Q-table.
5. After many episodes, it learns which actions are better in different situations.

### State Information

The agent considers:

- Player's current total.
- Dealer's visible card.
- Whether the player has a usable Ace.

### Rewards

| Game Result | Reward |
|---|---:|
| Win | +1 |
| Loss | -1 |
| Draw | 0 |

## 🛠️ Technologies Used

- Python
- Reinforcement Learning
- Q-Learning
- NumPy-free implementation
- JSON for saving the trained model
- Matplotlib for strategy visualization

## 📂 Project Structure

```text
AI-Blackjack-Player/
│
├── blackjack.py
├── main.py
├── README.md
├── .gitignore
│
└── tests/
    └── test_blackjack.py
```

## ⚙️ Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Blackjack-Player.git
```

### 2. Open the project folder

```bash
cd AI-Blackjack-Player
```

### 3. Run the program

```bash
python main.py
```

### 4. Train the AI

Select the training option from the program menu. The agent will learn by playing simulated Blackjack games.

## 📊 Learning Approach

The project uses an epsilon-greedy action-selection strategy:

- **Exploration:** The AI tries different actions to discover better strategies.
- **Exploitation:** The AI chooses the action with the highest learned Q-value.

The Q-table stores the estimated value of taking each action in a particular game state.

## ⚠️ Limitations

- The game uses an infinite-deck approximation.
- The dealer stands on soft 17.
- Splitting, doubling down, and insurance are not implemented.
- The learned strategy is based on the simplified game environment.

## 🎯 Future Improvements

- Implement Double Down and Split.
- Add card counting.
- Build a graphical user interface.
- Compare Q-Learning with Monte Carlo methods.
- Improve the agent through more advanced reinforcement learning techniques.

## 👨‍💻 Author

**Swayam Anup Shrikondwar**

B.Sc. Computer Science Student

[GitHub Profile](https://github.com/sasgoku)

---

*This project was developed as a practical exploration of Python programming, game simulation, and reinforcement learning.*
