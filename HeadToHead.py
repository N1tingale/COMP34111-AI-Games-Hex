import argparse
import importlib
import multiprocessing
import os
import sys
from collections import Counter

# --- IMPORT GAME MODULES ---
from src.Colour import Colour
from src.Game import Game
from src.Player import Player

# ==========================================
#   USER CONFIGURATION (EDIT ME HERE)
# ==========================================

DEFAULT_NUM_GAMES = 3  # Total games to play per iteration per iteration
DEFAULT_NUM_ITERATIONS = 8  # Number of iterations to run

# AGENT 1
# Format: "module.path ClassName"
DEFAULT_P1_STR = "agents.CrisOptimisedGroup35.Group_35 TournamentAgent"
DEFAULT_P1_NAME = "NEWAGENT"

# AGENT 2
# Format: "module.path ClassName"
DEFAULT_P2_STR = "agents.CrisOptimisedGroup35_copy.Group_35 TournamentAgent"
DEFAULT_P2_NAME = "OLDAGENT"

# ==========================================

DEFAULT_BOARD_SIZE = 11

def play_single_match(p1_str, p1_name, p2_str, p2_name, game_id):
    try:
        # Ensure imports work in subprocess
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # Parse "module.path ClassName"
        p1_path, p1_class_name = p1_str.split(" ")
        p2_path, p2_class_name = p2_str.split(" ")

        # Dynamic Import & RELOAD (Fixes dirty memory/global var cheating)
        mod1 = importlib.import_module(p1_path)
        importlib.reload(mod1) 
        mod2 = importlib.import_module(p2_path)
        importlib.reload(mod2)
        
        # Instantiate Agents
        agent1 = getattr(mod1, p1_class_name)(Colour.RED)
        agent2 = getattr(mod2, p2_class_name)(Colour.BLUE)

        player1 = Player(p1_name, agent1)
        player2 = Player(p2_name, agent2)

        game = Game(
            player1=player1, 
            player2=player2, 
            board_size=DEFAULT_BOARD_SIZE, 
            silent=True,
            logDest=os.devnull
        )
        
        result = game.run()
        winner = result['winner']

        # Fix Result Type Mismatch
        # Check if winner is a Player object and extract name, otherwise cast to str
        winner_name = winner.name if hasattr(winner, 'name') else str(winner)
        
        return winner_name

    except Exception as e:
        print(f"Exception in game: {e}")
        # Return error as string so the tally logic works
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Run concurrent Head-to-Head Hex games.")
    
    # Defaults are pulled from the constants at the top
    parser.add_argument("-n", "--num_games", type=int, default=DEFAULT_NUM_GAMES)
    parser.add_argument("-i", "--iterations", type=int, default=DEFAULT_NUM_ITERATIONS, help="Number of iterations to run (each iteration runs num_games in parallel)")
    parser.add_argument("-p1", "--player1", default=DEFAULT_P1_STR)
    parser.add_argument("-p1Name", "--player1Name", default=DEFAULT_P1_NAME)
    parser.add_argument("-p2", "--player2", default=DEFAULT_P2_STR)
    parser.add_argument("-p2Name", "--player2Name", default=DEFAULT_P2_NAME)
    
    args = parser.parse_args()

    # --- SETUP ---
    print(f"\n--- HEX HEAD-TO-HEAD TOURNAMENT ---")
    print(f"Agent 1: {args.player1Name}")
    print(f"Agent 2: {args.player2Name}")
    print(f"Games per Iteration: {args.num_games}")
    print(f"Iterations: {args.iterations} (Total Games: {args.num_games * args.iterations})")
    print(f"Running serially across iterations, parallel within each...\n")

    # Initialize total counters
    total_wins_p1 = 0
    total_wins_p2 = 0
    total_errors = 0

    # --- RUN ITERATIONS ---
    for iter_num in range(1, args.iterations + 1):
        print(f"Iteration {iter_num}/{args.iterations}...")

        # Alternate which agent is P1 (first player) for fairness
        if iter_num % 2 == 1:
            current_p1_str, current_p1_name = args.player1, args.player1Name
            current_p2_str, current_p2_name = args.player2, args.player2Name
            print(f"  {args.player1Name} as P1 (Red), {args.player2Name} as P2 (Blue)")
        else:
            current_p1_str, current_p1_name = args.player2, args.player2Name
            current_p2_str, current_p2_name = args.player1, args.player1Name
            print(f"  {args.player2Name} as P1 (Red), {args.player1Name} as P2 (Blue)")

        # --- PREPARE TASKS ---
        # Alternate colors for fairness within the iteration
        tasks = []
        for i in range(args.num_games):
            if i % 2 == 0:
                # Game i: Current P1 is Red, Current P2 is Blue
                tasks.append((
                    current_p1_str, current_p1_name, 
                    current_p2_str, current_p2_name, 
                    i+1
                ))
            else:
                # Game i+1: Current P2 is Red, Current P1 is Blue (Swap roles)
                tasks.append((
                    current_p2_str, current_p2_name, 
                    current_p1_str, current_p1_name, 
                    i+1
                ))

        # --- RUN CONCURRENTLY ---
        with multiprocessing.Pool() as pool:
            results = pool.starmap(play_single_match, tasks)

        # --- TALLY THIS ITERATION ---
        tally = Counter(results)
        
        wins_p1 = tally[args.player1Name]
        wins_p2 = tally[args.player2Name]
        errors = sum(1 for r in results if "ERROR" in r)

        total_wins_p1 += wins_p1
        total_wins_p2 += wins_p2
        total_errors += errors

        print(f"  Iteration {iter_num} Results: {args.player1Name} {wins_p1}, {args.player2Name} {wins_p2}, Errors {errors}")

    # --- FINAL RESULTS ---
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS (Total Games: {args.num_games * args.iterations})")
    print("=" * 50)
    print(f"{args.player1Name:>15} Total Wins: {total_wins_p1}")
    print(f"{args.player2Name:>15} Total Wins: {total_wins_p2}")
    
    if total_errors > 0:
        print(f"{'Total Errors':>15}: {total_errors}")

    print("=" * 50)
    
    if total_wins_p1 > total_wins_p2:
        print(f"🏆 OVERALL WINNER: {args.player1Name}")
    elif total_wins_p2 > total_wins_p1:
        print(f"🏆 OVERALL WINNER: {args.player2Name}")
    else:
        print(f"🤝 OVERALL DRAW")
    print("=" * 50)

if __name__ == "__main__":
    main()