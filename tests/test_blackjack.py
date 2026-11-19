import tempfile
import unittest
from pathlib import Path

from blackjack import evaluate_agent, hand_value, load_q_table, save_q_table, state_key, train_agent


class BlackjackTests(unittest.TestCase):
    def test_ace_counts_as_eleven_when_possible(self):
        self.assertEqual(hand_value([11, 7]), (18, True))

    def test_ace_changes_to_one_when_needed(self):
        self.assertEqual(hand_value([11, 9, 5]), (15, False))

    def test_two_aces(self):
        self.assertEqual(hand_value([11, 11, 5]), (17, True))

    def test_state_key(self):
        self.assertEqual(state_key((18, 10, True)), "18|10|1")

    def test_training_runs(self):
        table, rewards = train_agent(episodes=250, seed=7)
        self.assertTrue(table)
        self.assertEqual(len(rewards), 250)

    def test_model_save_and_load(self):
        table, _ = train_agent(episodes=30, seed=5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            save_q_table(table, str(path))
            self.assertEqual(load_q_table(str(path)), table)

    def test_evaluation_counts_each_round(self):
        table, _ = train_agent(episodes=100, seed=3)
        result = evaluate_agent(table, episodes=50, seed=4)
        self.assertEqual(result["wins"] + result["losses"] + result["pushes"], 50)


if __name__ == "__main__":
    unittest.main()
