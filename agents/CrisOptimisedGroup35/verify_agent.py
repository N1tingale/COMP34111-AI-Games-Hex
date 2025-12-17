import time
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# --- Import your agent ---
# Ensure Group_35.py is in the same directory
from Group_35 import TournamentAgent, board_to_numpy, numpy_to_tensor
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

def board_to_tensor_compatibility(board, colour):
    """Wrapper to make the optimized agent compatible with this test"""
    # Convert Object Board -> Numpy
    grid = board_to_numpy(board)
    # Convert Enum -> Int (Red=1, Blue=2)
    player_int = 1 if colour == Colour.RED else 2
    # Convert Numpy -> Tensor
    return numpy_to_tensor(grid, player_int)

def test_tensor_encoding():
    print("\n--- Test 1: Tensor Encoding (Orientation Check) ---")
    b = Board(BOARD_SIZE)
    b.set_tile_colour(0, 0, Colour.RED)
    
    # Check RED (Vertical)
    t_red = board_to_tensor_compatibility(b, Colour.RED)
    # Plane 2 should be ALL 1.0 (Red = Vertical)
    pass_red = torch.all(t_red[0, 2] == 1.0)
    
    # Check BLUE (Horizontal)
    t_blue = board_to_tensor_compatibility(b, Colour.BLUE)
    # Plane 2 should be ALL 0.0 (Blue = Horizontal)
    pass_blue = torch.all(t_blue[0, 2] == 0.0)

    if pass_red and pass_blue:
        print_result("Tensor Encoding", True, "Agent knows Red=Vertical, Blue=Horizontal.")
    else:
        print_result("Tensor Encoding", False, "Agent is confused about orientation!")

def test_swap_mechanics():
    print("\n--- Test 2: Swap Mechanics (Internal Numpy Logic) ---")
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)
    # Opponent (Red) played at (0, 5) - Top Edge
    b.set_tile_colour(0, 5, Colour.RED)
    
    # 1. Convert to internal Numpy representation
    grid = board_to_numpy(b)
    
    # 2. Force apply Swap using the new Optimized Method
    agent.mcts._apply_move_numpy(grid, Move(-1, -1), 2)
    
    # 3. Check Logic: The stone SHOULD move to the transpose (5,0)
    # Original spot (0,5) should be empty
    original_spot_empty = (grid[0, 5] == 0)
    # Transposed spot (5,0) should be BLUE (2)
    transposed_spot_filled = (grid[5, 0] == 2)
    
    if original_spot_empty and transposed_spot_filled:
        print_result("Swap Physics", True, "Agent correctly TRANSPOSED the stone for the model.")
    else:
        print_result("Swap Physics", False, f"Swap logic failed. Grid state: (0,5)={grid[0,5]}, (5,0)={grid[5,0]}")

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

def test_swap_ghost_stone_fix():
    print("\n--- Test 10: GHOST STONE REPLAY (The 'Double Play' Bug) ---")
    # Scenario: 
    # 1. We (Player 1) previously played at (0, 4).
    # 2. Opponent (Player 2) SWAPPED.
    # 3. Game Engine gave the stone at (0, 4) to Opponent (Red).
    # 4. We are now BLUE. We must NOT play at (0, 4) again.
    
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)
    
    # Setup the board exactly as it looks after a swap
    # The stone played by P1 stays at (0,4) but now belongs to RED (Opponent)
    b.set_tile_colour(0, 4, Colour.RED)
    
    print("   Scenario: Stone at (0,4) belongs to Red. Opponent just Swapped.")
    print("   Checking if Agent (Blue) realizes (0,4) is occupied...")

    try:
        # We pass Move(-1, -1) to signal the swap happened
        move = agent.make_move(turn=2, board=b, opp_move=Move(-1, -1))
        
        # FAILURE CONDITION: Agent tries to play (0, 4) again
        if move.x == 0 and move.y == 4:
            print_result("Ghost Replay", False, "CRITICAL: Agent tried to play (0,4) which is occupied! (Ghost Stone Bug)")
        # FAILURE CONDITION: Agent tries to play (4, 0) thinking it's the same spot
        elif move.x == 4 and move.y == 0 and b.tiles[4][0].colour is not None:
             print_result("Ghost Replay", False, "Agent got confused and played occupied (4,0).")
        else:
            print_result("Ghost Replay", True, f"Agent safely played {move}. Avoided the ghost stone.")
            
    except Exception as e:
        print_result("Ghost Replay", False, f"Crashed during thought process: {e}")

# Don't forget to add 'test_ghost_stone_replay_crash()' to the runner at the bottom!

# ==========================================
# 3. STABILITY & EDGE CASES
# ==========================================

def test_opening_move():
    print("\n--- Test 7: OPENING MOVE (Empty Board) ---")
    agent = TournamentAgent(Colour.RED)
    b = Board(BOARD_SIZE)
    
    start = time.time()
    try:
        move = agent.make_move(turn=1, board=b, opp_move=None)
        duration = time.time() - start
        
        # Check validity
        if move.x >= 0 and move.x < BOARD_SIZE and move.y >= 0 and move.y < BOARD_SIZE:
            print_result(f"Opening Move", True, f"Played {move} in {duration:.2f}s")
        else:
            print_result(f"Opening Move", False, f"Invalid move returned: {move}")
            
    except Exception as e:
        print_result("Opening Move", False, f"Crashed on Turn 1: {e}")

def test_panic_mode():
    print("\n--- Test 8: PANIC MODE (Low Time Budget) ---")
    agent = TournamentAgent(Colour.RED)
    b = Board(BOARD_SIZE)
    
    # Simulate we have used almost all our time
    agent.time_budget = 290.0
    agent.time_used = 288.0 # Only 2 seconds left!
    
    start = time.time()
    try:
        move = agent.make_move(turn=50, board=b, opp_move=Move(0,0))
        duration = time.time() - start
        
        if duration < 2.5: # Should return very quickly
            print_result("Panic Mode", True, f"Agent played safely in {duration:.2f}s with <2s budget.")
        else:
            print_result("Panic Mode", False, f"Agent timed out! Took {duration:.2f}s with <2s budget.")
            
    except Exception as e:
        print_result("Panic Mode", False, f"Crashed under pressure: {e}")

def test_endgame_fill():
    print("\n--- Test 9: ENDGAME (1 Empty Spot Left) ---")
    agent = TournamentAgent(Colour.RED)
    b = Board(BOARD_SIZE)
    
    # Fill the board completely except (5,5)
    last_empty = (5,5)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r,c) != last_empty:
                # Fill with alternating colors
                color = Colour.RED if (r+c)%2==0 else Colour.BLUE
                b.set_tile_colour(r, c, color)
    
    print("   Scenario: Board full except (5,5). Agent MUST play (5,5).")
    
    try:
        move = agent.make_move(turn=120, board=b, opp_move=Move(0,0))
        
        if move.x == 5 and move.y == 5:
            print_result("Endgame", True, "Agent played the only remaining legal move.")
        else:
            print_result("Endgame", False, f"Agent failed to find the last move. Played {move}.")
            
    except Exception as e:
        print_result("Endgame", False, f"Crashed on full board: {e}")

def test_illegal_turn_3_swap():
    print("\n--- Test 11: ILLEGAL SWAP CHECK (Turn 3 Bug) ---")
    # Scenario: 
    # The board has exactly 1 stone (e.g., Opponent swapped last turn).
    # A naive agent sees 1 stone and thinks "I can swap!".
    # A correct agent sees Turn=3 and knows "Swap is illegal".
    
    agent = TournamentAgent(Colour.RED)
    b = Board(BOARD_SIZE)
    
    # Setup a board with 1 stone (e.g., Red at 0,4)
    # This simulates the state after P2 swapped a stone to (0,4)
    b.set_tile_colour(0, 4, Colour.RED)
    
    print("   Scenario: Turn 3. Board has 1 stone. Swap is ILLEGAL.")
    
    try:
        # We tell the agent it is Turn 3
        move = agent.make_move(turn=3, board=b, opp_move=Move(-1, -1))
        
        if move.is_swap():
            print_result("Turn 3 Check", False, "CRITICAL: Agent tried to Swap on Turn 3! (Illegal Move)")
        else:
            print_result("Turn 3 Check", True, f"Agent correctly played a normal move: {move}")
            
    except Exception as e:
        print_result("Turn 3 Check", False, f"Crashed: {e}")

def test_edge_stability():
    print("\n--- Test 12: EDGE CASE STABILITY ---")
    # Scenario: Force the agent to evaluate edge positions.
    # We fill the center, leaving only the edges open.
    
    agent = TournamentAgent(Colour.BLUE)
    b = Board(BOARD_SIZE)
    
    # Fill the inner 9x9 square, leaving the outer rim empty
    for r in range(1, BOARD_SIZE-1):
        for c in range(1, BOARD_SIZE-1):
            b.set_tile_colour(r, c, Colour.RED)
            
    print("   Scenario: Center is full. Agent MUST play on the edge.")
    
    try:
        move = agent.make_move(turn=50, board=b, opp_move=Move(5,5))
        
        # Check if move is on the boundary
        is_edge = (move.x == 0 or move.x == BOARD_SIZE-1 or 
                   move.y == 0 or move.y == BOARD_SIZE-1)
                   
        if is_edge:
            print_result("Edge Stability", True, f"Agent comfortably played on the edge: {move}")
        else:
            # This would imply it played on an occupied center tile
            print_result("Edge Stability", False, f"Agent played invalid move {move} (likely occupied).")

    except Exception as e:
        print_result("Edge Stability", False, f"Crashed looking at edges: {e}")

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
        test_swap_ghost_stone_fix()
        test_opening_move()
        test_panic_mode()
        test_endgame_fill()
        test_illegal_turn_3_swap()
        test_edge_stability()
        
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== Verification Complete ===")