"""Blackjack environment and Q-learning agent."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STAND = 0
HIT = 1
CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
State = Tuple[int, int, bool]
QTable = Dict[str, List[float]]


def draw_card(rng=None) -> int:
    """Draw from an infinite-deck approximation (cards sampled with replacement)."""
    return (rng or random).choice(CARD_VALUES)


def hand_value(cards: List[int]) -> Tuple[int, bool]:
    """Return the best total and whether an ace is still counted as 11."""
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


def get_state(player_cards: List[int], dealer_up_card: int) -> State:
    total, usable_ace = hand_value(player_cards)
    return total, dealer_up_card, usable_ace


def dealer_turn(dealer_cards: List[int], rng=None) -> List[int]:
    """Dealer hits below 17 and stands on all 17s, including soft 17."""
    while hand_value(dealer_cards)[0] < 17:
        dealer_cards.append(draw_card(rng))
    return dealer_cards


def get_reward(player_cards: List[int], dealer_cards: List[int]) -> float:
    player_total, _ = hand_value(player_cards)
    dealer_total, _ = hand_value(dealer_cards)
    if player_total > 21:
        return -1.0
    if dealer_total > 21 or player_total > dealer_total:
        return 1.0
    if player_total < dealer_total:
        return -1.0
    return 0.0


def state_key(state: State) -> str:
    player_total, dealer_card, usable_ace = state
    return f"{player_total}|{dealer_card}|{int(usable_ace)}"


def q_values(q_table: QTable, state: State) -> List[float]:
    key = state_key(state)
    if key not in q_table:
        q_table[key] = [0.0, 0.0]  # [Q(stand), Q(hit)]
    return q_table[key]


def choose_action(q_table: QTable, state: State, epsilon: float = 0.0, rng=None) -> int:
    """Choose with epsilon-greedy exploration."""
    generator = rng or random
    if generator.random() < epsilon:
        return generator.choice([STAND, HIT])
    values = q_values(q_table, state)
    if values[STAND] == values[HIT]:
        return generator.choice([STAND, HIT])
    return STAND if values[STAND] > values[HIT] else HIT


def play_episode(q_table: QTable, alpha: float = 0.1, gamma: float = 1.0,
                 epsilon: float = 0.0, rng=None, learn: bool = False,
                 show: bool = False) -> float:
    """Play one round, optionally updating the agent's Q-values."""
    generator = rng or random
    player = [draw_card(generator), draw_card(generator)]
    dealer = [draw_card(generator), draw_card(generator)]
    dealer_up = dealer[0]

    # In this simplified rule set, totals below 12 are automatic hits.
    while hand_value(player)[0] < 12:
        player.append(draw_card(generator))

    if show:
        print(f"\nAI cards: {player} (total {hand_value(player)[0]})")
        print(f"Dealer shows: {dealer_up} and [hidden]")

    while True:
        state = get_state(player, dealer_up)
        action = choose_action(q_table, state, epsilon, generator)
        if show:
            print("AI decision:", "HIT" if action == HIT else "STAND")

        if action == HIT:
            player.append(draw_card(generator))
            total, _ = hand_value(player)
            if show:
                print(f"AI draws {player[-1]}; new total: {total}")
            if total > 21:
                if learn:
                    values = q_values(q_table, state)
                    values[HIT] += alpha * (-1.0 - values[HIT])
                if show:
                    print("AI busts. Dealer wins.")
                return -1.0
            if learn:
                next_state = get_state(player, dealer_up)
                current = q_values(q_table, state)
                future = q_values(q_table, next_state)
                target = gamma * max(future)
                current[HIT] += alpha * (target - current[HIT])
        else:
            if show:
                print(f"AI stands at {hand_value(player)[0]}")
                print(f"Dealer's hidden card: {dealer[1]}")
            dealer_turn(dealer, generator)
            reward = get_reward(player, dealer)
            if show:
                dealer_total, _ = hand_value(dealer)
                print(f"Dealer's final hand: {dealer} (total {dealer_total})")
                print("AI wins!" if reward > 0 else "Dealer wins." if reward < 0 else "Push (tie).")
            if learn:
                values = q_values(q_table, state)
                values[STAND] += alpha * (reward - values[STAND])
            return reward


def train_agent(episodes: int = 100_000, alpha: float = 0.1, gamma: float = 1.0,
                epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                seed: Optional[int] = 42) -> Tuple[QTable, List[float]]:
    """Train through simulated hands and return (Q-table, rewards)."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = random.Random(seed)
    table: QTable = {}
    rewards: List[float] = []
    report_every = max(1, episodes // 10)
    for episode in range(episodes):
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * fraction
        rewards.append(play_episode(table, alpha, gamma, epsilon, rng, learn=True))
        if (episode + 1) % report_every == 0 or episode == episodes - 1:
            print(f"\rTraining: {episode + 1:,}/{episodes:,} rounds", end="")
    print("\nTraining complete.")
    return table, rewards


def evaluate_agent(q_table: QTable, episodes: int = 10_000,
                   seed: Optional[int] = 123) -> dict:
    """Evaluate the current policy; evaluation does not update learned values."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    rng = random.Random(seed)
    wins = losses = pushes = 0
    for _ in range(episodes):
        result = play_episode(q_table, rng=rng, learn=False)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            pushes += 1
    return {
        "rounds": episodes,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_percentage": round(wins / episodes * 100, 2),
        "loss_percentage": round(losses / episodes * 100, 2),
        "push_percentage": round(pushes / episodes * 100, 2),
    }


def save_q_table(q_table: QTable, filename: str = "blackjack_q_table.json") -> None:
    Path(filename).write_text(json.dumps(q_table, indent=2), encoding="utf-8")


def load_q_table(filename: str = "blackjack_q_table.json") -> QTable:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Model '{filename}' not found. Train the AI first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The model file has an invalid format.")
    return data


def strategy_chart(q_table: QTable) -> str:
    """Display learned Hit/Stand choices for player totals and dealer cards."""
    dealer_cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    lines = ["LEARNED POLICY (H = Hit, S = Stand)",
             "Player | " + " ".join(f"{x:>2}" for x in dealer_cards), "-" * 43]
    for usable_ace in (False, True):
        lines.append("Usable ace: " + ("Yes" if usable_ace else "No"))
        for total in range(12, 22):
            marks = []
            for dealer_card in dealer_cards:
                values = q_values(q_table, (total, dealer_card, usable_ace))
                marks.append("?" if values[0] == values[1] else "S" if values[0] > values[1] else "H")
            lines.append(f"{total:>6} | " + " ".join(f"{x:>2}" for x in marks))
    return "\n".join(lines)
