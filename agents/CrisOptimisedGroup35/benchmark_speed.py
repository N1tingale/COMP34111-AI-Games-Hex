import time
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from Group_35 import TournamentAgent


def benchmark_agent():
    print("--- 🚀 STARTING SPEED BENCHMARK ---")

    # 1. Initialize Agent
    agent = TournamentAgent(Colour.RED)
    print(f"Model Device: {next(agent.model.parameters()).device}")

    # 2. Setup Board
    board = Board(11)

    # 3. Warmup (JIT compilation happens here)
    print("Warmup move (triggering JIT compilation)...")
    agent.time_budget = 10.0  # Give it budget
    # Turn 1 passed implicitly here via make_move
    agent.make_move(1, board, None)

    # 4. Stress Test
    print("\nStarting Stress Test (5.0s fixed time)...")

    # Reset agent state for a clean move
    agent.time_used = 0
    agent.time_budget = 1000.0

    agent.mcts.root = None
    agent.mcts.transposition_table = {}

    start = time.time()
    # FIX: Added turn_number=1 argument
    best_move = agent.mcts.search(board, Colour.RED, time_limit=5.0, turn_number=1)
    end = time.time()

    duration = end - start
    
    if agent.mcts.root is not None:
        total_sims = agent.mcts.root.visit_count
        sps = total_sims / duration
    else:
        total_sims = 0
        sps = 0

    print(f"\n--- 📊 RESULTS ---")
    print(f"Total Time:       {duration:.4f} seconds")
    print(f"Total Simulations: {total_sims}")
    print(f"Performance:       {sps:.1f} Sims/s")
    print("------------------")

    if sps < 10:
        print("❌ Speed is Low. Optimizations might not be engaging.")
    elif sps < 15:
        print("⚠️ Speed is Moderate. Python overhead is still present.")
    else:
        print("✅ Speed is Excellent! Vectorization & JIT are working.")


if __name__ == "__main__":
    benchmark_agent()