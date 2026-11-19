"""Menu-driven terminal application for the AI Blackjack Player."""
from blackjack import evaluate_agent, load_q_table, play_episode, save_q_table, strategy_chart, train_agent

MODEL_FILE = "blackjack_q_table.json"


def ask_number(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        number = int(raw)
        if number < 1:
            raise ValueError
        return number
    except ValueError:
        print("Enter a whole number greater than zero.")
        return ask_number(prompt, default)


def get_model():
    try:
        return load_q_table(MODEL_FILE)
    except (FileNotFoundError, ValueError):
        print("No trained model found. Training a new AI (100,000 rounds)...")
        table, _ = train_agent(episodes=100_000)
        save_q_table(table, MODEL_FILE)
        print(f"Model saved as {MODEL_FILE}.")
        return table


def main():
    print("=" * 54)
    print("              AI BLACKJACK PLAYER")
    print("       Reinforcement Learning with Q-Learning")
    print("=" * 54)
    while True:
        print("\n1. Train the AI")
        print("2. Watch the AI play")
        print("3. Evaluate the AI")
        print("4. View the learned strategy")
        print("5. Exit")
        choice = input("\nChoose an option (1-5): ").strip()
        if choice == "1":
            episodes = ask_number("Training rounds", 100_000)
            table, _ = train_agent(episodes=episodes)
            save_q_table(table, MODEL_FILE)
            print(f"Trained model saved as '{MODEL_FILE}'.")
        elif choice == "2":
            table = get_model()
            input("\nPress Enter to watch one round...")
            play_episode(table, show=True, learn=False)
        elif choice == "3":
            table = get_model()
            episodes = ask_number("Evaluation rounds", 10_000)
            result = evaluate_agent(table, episodes)
            print("\nEVALUATION RESULTS")
            print("-" * 30)
            print(f"Rounds: {result['rounds']:,}")
            print(f"Wins:   {result['wins']:,} ({result['win_percentage']}%)")
            print(f"Losses: {result['losses']:,} ({result['loss_percentage']}%)")
            print(f"Pushes: {result['pushes']:,} ({result['push_percentage']}%)")
            print("Win percentage is not the same as profitability in a real casino.")
        elif choice == "4":
            print("\n" + strategy_chart(get_model()))
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Select 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
