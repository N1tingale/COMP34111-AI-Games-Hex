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

DEFAULT_NUM_GAMES = 10  # Total games to play

# AGENT 1
# Format: "module.path ClassName"
DEFAULT_P1_STR = "agents.CrisOptimisedGroup35.Group_35 TournamentAgent"
DEFAULT_P1_NAME = "CrisGroup35"

# AGENT 2
# Format: "module.path ClassName"
DEFAULT_P2_STR = "agents.Group35.Group_35 TournamentAgent"
DEFAULT_P2_NAME = "KareemGroup35"

# ==========================================

DEFAULT_BOARD_SIZE = 11

def play_single_match(p1_str, p1_name, p2_str, p2_name, game_id):
    """
    Runs a single game of Hex.
    p1 starts as RED (First Player).
    p2 starts as BLUE (Second Player).
    """
    try:
        # Parse "module.path ClassName" string
        p1_path, p1_class_name = p1_str.split(" ")
        p2_path, p2_class_name = p2_str.split(" ")

        # Dynamic Import
        mod1 = importlib.import_module(p1_path)
        mod2 = importlib.import_module(p2_path)
        
        # Instantiate Agents
        agent1 = getattr(mod1, p1_class_name)(Colour.RED)
        agent2 = getattr(mod2, p2_class_name)(Colour.BLUE)

        player1 = Player(p1_name, agent1)
        player2 = Player(p2_name, agent2)

        # Create and Run Game
        # We pass logDest=None or os.devnull to avoid creating files
        game = Game(
            player1=player1, 
            player2=player2, 
            board_size=DEFAULT_BOARD_SIZE, 
            silent=True,        # Keep console clean
            logDest=os.devnull  # Discard logs
        )
        
        result = game.run()
        return result['winner']

    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Run concurrent Head-to-Head Hex games.")
    
    # Defaults are pulled from the constants at the top
    parser.add_argument("-n", "--num_games", type=int, default=DEFAULT_NUM_GAMES)
    parser.add_argument("-p1", "--player1", default=DEFAULT_P1_STR)
    parser.add_argument("-p1Name", "--player1Name", default=DEFAULT_P1_NAME)
    parser.add_argument("-p2", "--player2", default=DEFAULT_P2_STR)
    parser.add_argument("-p2Name", "--player2Name", default=DEFAULT_P2_NAME)
    
    args = parser.parse_args()

    # --- SETUP ---
    print(f"\n--- HEX HEAD-TO-HEAD TOURNAMENT ---")
    print(f"Agent 1: {args.player1Name}")
    print(f"Agent 2: {args.player2Name}")
    print(f"Games:   {args.num_games} (Simulating concurrently...)\n")

    # --- PREPARE TASKS ---
    # Alternate colors for fairness
    tasks = []
    for i in range(args.num_games):
        if i % 2 == 0:
            # Game i: Agent 1 is Red, Agent 2 is Blue
            tasks.append((
                args.player1, args.player1Name, 
                args.player2, args.player2Name, 
                i+1
            ))
        else:
            # Game i+1: Agent 2 is Red, Agent 1 is Blue (Swap roles)
            tasks.append((
                args.player2, args.player2Name, 
                args.player1, args.player1Name, 
                i+1
            ))

    # --- RUN CONCURRENTLY ---
    with multiprocessing.Pool() as pool:
        results = pool.starmap(play_single_match, tasks)

    # --- RESULTS ---
    tally = Counter(results)
    
    wins_p1 = tally[args.player1Name]
    wins_p2 = tally[args.player2Name]
    errors = sum(1 for r in results if "ERROR" in r)

    print("-" * 40)
    print(f"FINAL RESULTS ({args.num_games} Games)")
    print("-" * 40)
    print(f"{args.player1Name:>15} Wins: {wins_p1}")
    print(f"{args.player2Name:>15} Wins: {wins_p2}")
    
    if errors > 0:
        print(f"{'Errors':>15}: {errors}")
        print("\nFirst Error:", [r for r in results if "ERROR" in r][0])

    print("-" * 40)
    
    if wins_p1 > wins_p2:
        print(f"🏆 WINNER: {args.player1Name}")
    elif wins_p2 > wins_p1:
        print(f"🏆 WINNER: {args.player2Name}")
    else:
        print(f"🤝 DRAW")
    print("-" * 40)

if __name__ == "__main__":
    main()