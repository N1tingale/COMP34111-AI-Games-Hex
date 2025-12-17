import time
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- Import your agent ---
# Ensure Group_35.py is in the same directory
from Group_35 import TournamentAgent, board_to_tensor
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# --- Configuration ---
BOARD_SIZE = 11

def print_result(name, passed, message=""):
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    print(f"{icon} [{status}] {name}: {message}")
    if not passed:
        print(f"   !!! Critical Failure in {name} !!!")

# ==========================================
# 1. CORE LOGIC (Tensor & Swap)
# ==========================================

def test_tensor_encoding():
    print("\n--- Test 1: Tensor Encoding (Orientation Check) ---")
    b = Board(BOARD_SIZE)
    b.set_tile_colour(0, 0, Colour.RED)
    
    # Check RED (Vertical)
    t_red = board_to_tensor(b, Colour.RED)
    # Plane 2 should be ALL 1.0 (Red = Vertical)
    pass_red = torch.all(t_red[0, 2] == 1.0)
    
    # Check BLUE (Horizontal)
    t_blue = board_to_tensor(b, Colour.BLUE)
    # Plane 2 should be ALL 0.0 (Blue = Horizontal)
    pass_blue = torch.all(t_blue[0, 2] == 0.0)

    if pass_red and pass_blue:
        print_result("Tensor Encoding", True, "Agent knows Red=Vertical, Blue=Horizontal.")
    else:
        print_result("Tensor Encoding", False, "Agent is confused about orientation!")

def test_swap_mechanics():
    print("\n--- Test 2: Swap Mechanics (Internal Logic) ---")
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)
    # Opponent (Red) played at (0, 5)
    b.set_tile_colour(0, 5, Colour.RED)
    
    # Force apply Swap
    agent.mcts._apply_move(b, Move(-1, -1), Colour.BLUE)
    
    # Check Transpose: (0,5) empty, (5,0) Blue
    pass_swap = (b.tiles[0][5].colour is None) and (b.tiles[5][0].colour == Colour.BLUE)
    
    if pass_swap:
        print_result("Swap Physics", True, "Agent correctly transposed the board.")
    else:
        print_result("Swap Physics", False, "Swap logic failed (Did not transpose).")

# ==========================================
# 2. DETERMINISTIC "MUST PLAY" SCENARIOS
# ==========================================

def test_critical_win():
    print("\n--- Test 3: CRITICAL WIN (One move to win) ---")
    # Scenario: Red has a nearly complete vertical chain.
    # It is missing exactly ONE stone at (5, 5) to connect Top to Bottom.
    agent = TournamentAgent(Colour.RED)
    b = Board(BOARD_SIZE)
    
    # Top Half Chain
    for r in range(0, 5): b.set_tile_colour(r, 5, Colour.RED)
    # Bottom Half Chain
    for r in range(6, 11): b.set_tile_colour(r, 5, Colour.RED)
    
    print("   Scenario: Gap at (5,5). If Red plays there, Red wins instantly.")
    
    # Agent plays
    move = agent.make_move(turn=20, board=b, opp_move=None)
    
    if move.x == 5 and move.y == 5:
        print_result("Critical Win", True, "Agent found the winning move (5,5).")
    else:
        print_result("Critical Win", False, f"Agent missed the win! Played {move} instead of (5,5).")

def test_critical_block():
    print("\n--- Test 4: CRITICAL BLOCK (One move to die) ---")
    # Scenario: We are BLUE. Red is about to win.
    # Red has a chain that needs only (5, 5) to win.
    # We MUST play (5, 5) to block.
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)
    
    # Red's Threatening Chain
    for r in range(0, 5): b.set_tile_colour(r, 5, Colour.RED)
    for r in range(6, 11): b.set_tile_colour(r, 5, Colour.RED)
    
    print("   Scenario: Red will win at (5,5). Blue MUST block there.")
    
    # Agent plays
    move = agent.make_move(turn=20, board=b, opp_move=None)
    
    if move.x == 5 and move.y == 5:
        print_result("Critical Block", True, "Agent blocked the threat at (5,5).")
    else:
        print_result("Critical Block", False, f"Agent failed to block! Played {move} and lost.")

def test_must_swap():
    print("\n--- Test 5: STRATEGIC SWAP (Must steal winning move) ---")
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)

    # Red plays the strongest possible move: Center (5,5)
    b.set_tile_colour(5, 5, Colour.RED)
    print("   Scenario: Red opens at (5,5) (Center). Blue MUST swap.")

    # 1. Ask for move (Turn 2)
    # We use Turn 2 because Turn 1 is Red's move.
    move = agent.make_move(turn=2, board=b, opp_move=Move(5,5))

    if move.is_swap():
        print_result("Swap Decision", True, "Agent correctly chose to SWAP.")
    else:
        print_result("Swap Decision", False, f"Agent failed to swap! Played {move} against center opening.")
        return # Stop if failed

    # 2. Simulate Game Engine Logic (Role Swap)
    print("   ... Simulating Game Engine Role Swap (Blue -> Red) ...")
    agent.colour = Colour.RED # Game.py does this automatically when swap happens
    
    # The board remains as is (Red stone at 5,5). 
    # The agent is now Red. 
    # It is technically the opponent's turn now (Blue's turn).
    # Let's assume Blue plays a random move at (0,0).
    b.set_tile_colour(0, 0, Colour.BLUE)
    
    # 3. Ask for next move (Turn 4 - Red's turn)
    print("   ... Asking for next move as RED ...")
    move_after_swap = agent.make_move(turn=4, board=b, opp_move=Move(0,0))

    # Check validity
    if not move_after_swap.is_swap() and b.tiles[move_after_swap.x][move_after_swap.y].colour is None:
         print_result("Post-Swap Play", True, f"Agent played valid Red move {move_after_swap} after swapping.")
    else:
         print_result("Post-Swap Play", False, f"Agent confused after swap. Played {move_after_swap}.")

# ==========================================
# RUNNER
# ==========================================

if __name__ == "__main__":
    print("=== Hex Agent Strict Verification Suite ===")
    
    try:
        test_tensor_encoding()
        test_swap_mechanics()
        test_critical_win()
        test_critical_block()
        test_must_swap()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== Verification Complete ===")