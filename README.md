# AI Blackjack Player

A beginner-friendly Artificial Intelligence project that learns to play a simplified game of Blackjack using **Q-learning**, a Reinforcement Learning algorithm.

The agent practises simulated rounds, updates its Q-values from rewards, and gradually learns when to **Hit** or **Stand**.

## Features

- Train the agent through simulated Blackjack rounds.
- Use Q-learning and epsilon-greedy exploration.
- Watch the trained AI play against the dealer.
- Evaluate the agent over thousands of rounds (wins, losses, pushes).
- Display a text chart of the learned strategy.
- Save and reload the learned Q-table as JSON.
- Includes unit tests.

## Technologies

Python 3, Reinforcement Learning, Q-learning, epsilon-greedy exploration, JSON, and `unittest`.

No external packages or paid APIs are required.

## Run the project

1. Extract the ZIP file.
2. Open a terminal inside the `blackjack_ai_agent` folder.
3. Run:

```bash
python main.py
```

On some Linux systems, use `python3 main.py`.

Choose **1** to train the AI (start with 100,000 rounds). The learned model is saved in `blackjack_q_table.json`. Then choose **2** to watch it play, **3** to evaluate it, or **4** to view its learned strategy.

## How it works

### State
The agent observes its current hand total, the dealer's visible card, and whether it has a usable ace.

### Actions
The agent can **Hit** (draw a card) or **Stand** (stop drawing).

### Reward
A win gives `+1`, a loss gives `-1`, and a tie (push) gives `0`.

### Q-learning
The agent stores values for state/action pairs and updates them using the reward and the best estimated value of the next state:

`Q(s, a) = Q(s, a) + alpha * (reward + gamma * max Q(next_state) - Q(s, a))`

For a terminal action, the update uses the final reward because there is no next state.

### Exploration
Epsilon-greedy selection sometimes tries a random action. Epsilon decreases during training so the agent gradually relies more on its learned values.

## Rules and limitations

This is a simplified learning simulation:

- An infinite-deck approximation is used (cards are sampled with replacement).
- The dealer stands on all 17s, including soft 17.
- The AI chooses only Hit or Stand.
- Splitting, doubling down, insurance, betting, and casino payouts are not implemented.
- It is an educational simulation, not a gambling or profit-prediction system.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Future improvements

- Add a graphical interface with Pygame or Streamlit.
- Add Double Down and Split actions.
- Plot training and evaluation results with Matplotlib.
- Compare Q-learning against a basic Blackjack strategy.
- Add a human-versus-AI mode.

## CV description

**AI Blackjack Player using Reinforcement Learning | Python**

- Developed an AI Blackjack agent using Q-learning to learn Hit/Stand decisions through simulated gameplay.
- Implemented epsilon-greedy exploration and reward-based Q-value updates.
- Added training, model persistence, strategy visualization, and evaluation features to measure performance over simulated rounds.

Only list this project on your CV after you have run it, understood the main functions, and can explain Q-learning in an interview.
