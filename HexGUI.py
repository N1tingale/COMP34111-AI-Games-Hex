import tkinter as tk
from tkinter import scrolledtext
import sys
import subprocess
import threading
import re
import argparse


class HexBoardGUI:
    def __init__(self, root, player1=None, player2=None, board_size=11, p1_name="Alice", p2_name="Bob"):
        self.root = root
        self.root.title("Hex Game Visualizer - Live View")
        self.root.geometry("1200x900")
        
        self.player1 = player1
        self.player2 = player2
        self.board_size = board_size
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.current_board = None
        self.game_process = None
        
        # Create control frame
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        
        if player1 and player2:
            # Live game mode
            tk.Label(control_frame, text=f"Live Game: {p1_name} vs {p2_name}", 
                    font=("Arial", 14, "bold")).pack()
            tk.Button(control_frame, text="Start Game", command=self.start_live_game, 
                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
            self.status_label = tk.Label(control_frame, text="Ready to start", 
                                        font=("Arial", 10))
            self.status_label.pack()
        else:
            # Manual input mode
            tk.Label(control_frame, text="Paste board text here:").pack(anchor=tk.W)
            self.input_text = scrolledtext.ScrolledText(control_frame, height=8, width=100)
            self.input_text.pack(pady=5)
            tk.Button(control_frame, text="Visualize Board", command=self.visualize_board).pack(pady=5)
        
        # Create canvas for board display
        self.canvas = tk.Canvas(root, bg="white", width=1100, height=700)
        self.canvas.pack(pady=10, padx=10)
        
    def parse_board(self, text):
        """Parse the board text and extract the board state."""
        lines = text.strip().split('\n')
        board = []
        
        for line in lines:
            # Remove leading spaces and split by spaces
            line = line.strip()
            if not line:
                continue
                
            # Split the line and filter out empty strings
            cells = [c for c in line.split() if c]
            
            # Only add lines that look like board rows (contain R, B, or 0)
            if cells and any(c in ['R', 'B', '0'] for c in cells):
                board.append(cells)
        
        return board
    
    def visualize_board(self):
        """Draw the hex board on the canvas."""
        board_text = self.input_text.get("1.0", tk.END)
        board = self.parse_board(board_text)
        
        if not board:
            return
        
        self.canvas.delete("all")
        
        # Hexagon parameters
        hex_size = 25
        x_offset = 50
        y_offset = 50
        x_spacing = hex_size * 1.5
        y_spacing = hex_size * 1.732  # sqrt(3) for proper hex spacing
        
        # Draw hexagons
        for row_idx, row in enumerate(board):
            for col_idx, cell in enumerate(row):
                # Calculate position with offset for hex grid
                x = x_offset + col_idx * x_spacing + row_idx * (x_spacing / 2)
                y = y_offset + row_idx * y_spacing
                
                # Determine color
                if cell == 'R':
                    color = "#FF6B6B"  # Red
                    text_color = "white"
                elif cell == 'B':
                    color = "#4ECDC4"  # Blue/Cyan
                    text_color = "white"
                else:  # '0' or empty
                    color = "#F0F0F0"  # Light gray
                    text_color = "black"
                
                # Draw hexagon
                self.draw_hexagon(x, y, hex_size, color)
                
                # Draw cell content
                self.canvas.create_text(x, y, text=cell, fill=text_color, 
                                       font=("Arial", 10, "bold"))
        
        # Draw border labels
        board_size = len(board)
        
        # Top border (Red)
        self.canvas.create_text(x_offset - 20, y_offset - 20, 
                               text="RED", fill="#FF6B6B", 
                               font=("Arial", 12, "bold"))
        
        # Bottom border (Red)
        bottom_y = y_offset + (board_size - 1) * y_spacing
        self.canvas.create_text(x_offset + (board_size - 1) * (x_spacing / 2) + 20, 
                               bottom_y + 20, 
                               text="RED", fill="#FF6B6B", 
                               font=("Arial", 12, "bold"))
        
        # Left border (Blue)
        self.canvas.create_text(x_offset - 30, 
                               y_offset + (board_size / 2) * y_spacing, 
                               text="BLUE", fill="#4ECDC4", 
                               font=("Arial", 12, "bold"), angle=90)
        
        # Right border (Blue)
        right_x = x_offset + (board_size - 1) * x_spacing + (board_size - 1) * (x_spacing / 2)
        self.canvas.create_text(right_x + 30, 
                               y_offset + (board_size / 2) * y_spacing, 
                               text="BLUE", fill="#4ECDC4", 
                               font=("Arial", 12, "bold"), angle=90)
    
    def draw_hexagon(self, x, y, size, color):
        """Draw a hexagon at position (x, y) with given size and color."""
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = 3.14159 * angle_deg / 180
            point_x = x + size * round(3.14159 * angle_rad) / 3.14159
            point_y = y + size * round(angle_rad) / 3.14159
            
            # Simple approximation
            if i == 0:
                point_x, point_y = x + size, y
            elif i == 1:
                point_x, point_y = x + size/2, y + size * 0.866
            elif i == 2:
                point_x, point_y = x - size/2, y + size * 0.866
            elif i == 3:
                point_x, point_y = x - size, y
            elif i == 4:
                point_x, point_y = x - size/2, y - size * 0.866
            elif i == 5:
                point_x, point_y = x + size/2, y - size * 0.866
            
            points.append((point_x, point_y))
        
        # Flatten the points list
        flat_points = [coord for point in points for coord in point]
        
        # Draw filled hexagon
        self.canvas.create_polygon(flat_points, fill=color, outline="black", width=2)
    
    def start_live_game(self):
        """Start the game subprocess and begin reading output."""
        if self.game_process:
            return  # Game already running
        
        cmd = [
            "python", "Hex.py",
            "-p1", self.player1,
            "-p2", self.player2,
            "-p1Name", self.p1_name,
            "-p2Name", self.p2_name,
            "-b", str(self.board_size),
            "-v"
        ]
        
        self.status_label.config(text="Starting game...")
        
        # Start the game process
        self.game_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start reading output in a separate thread
        threading.Thread(target=self.read_game_output, daemon=True).start()
    
    def read_game_output(self):
        """Read game output and update the GUI."""
        board_lines = []
        capturing_board = False
        
        while True:
            line = self.game_process.stderr.readline()
            if not line:
                break
            
            line = line.rstrip()
            
            # Check for board markers
            if "Turn Ending Board:" in line or "Starting Board:" in line or "Final Board:" in line:
                capturing_board = True
                board_lines = []
                continue
            
            # Check for turn/game status
            if "Turn" in line and "player" in line:
                match = re.search(r'Turn (\d+): player (\w+)', line)
                if match:
                    turn_num = match.group(1)
                    player = match.group(2)
                    self.root.after(0, lambda t=turn_num, p=player: 
                                  self.status_label.config(text=f"Turn {t}: {p}'s move"))
            
            if "Game over" in line:
                self.root.after(0, lambda: self.status_label.config(text="Game Over!"))
            
            # Capture board lines
            if capturing_board:
                # Check if line looks like a board row
                if line and any(c in line for c in ['R', 'B', '0']):
                    board_lines.append(line)
                elif board_lines:
                    # We've captured a complete board
                    capturing_board = False
                    board_text = '\n'.join(board_lines)
                    self.root.after(0, lambda bt=board_text: self.update_board_from_text(bt))
                    board_lines = []
        
        self.root.after(0, lambda: self.status_label.config(text="Game finished"))
        self.game_process = None
    
    def update_board_from_text(self, board_text):
        """Update the board display from captured text."""
        board = self.parse_board(board_text)
        if board:
            self.visualize_board_data(board)
    
    def visualize_board_data(self, board):
        """Draw the hex board from board data."""
        if not board:
            return
        
        self.canvas.delete("all")
        
        # Hexagon parameters
        hex_size = 25
        x_offset = 50
        y_offset = 50
        x_spacing = hex_size * 1.5
        y_spacing = hex_size * 1.732  # sqrt(3) for proper hex spacing
        
        # Draw hexagons
        for row_idx, row in enumerate(board):
            for col_idx, cell in enumerate(row):
                # Calculate position with offset for hex grid
                x = x_offset + col_idx * x_spacing + row_idx * (x_spacing / 2)
                y = y_offset + row_idx * y_spacing
                
                # Determine color
                if cell == 'R':
                    color = "#FF6B6B"  # Red
                    text_color = "white"
                elif cell == 'B':
                    color = "#4ECDC4"  # Blue/Cyan
                    text_color = "white"
                else:  # '0' or empty
                    color = "#F0F0F0"  # Light gray
                    text_color = "black"
                
                # Draw hexagon
                self.draw_hexagon(x, y, hex_size, color)
                
                # Draw cell content
                self.canvas.create_text(x, y, text=cell, fill=text_color, 
                                       font=("Arial", 10, "bold"))
        
        # Draw border labels
        board_size = len(board)
        
        # Top border (Red)
        self.canvas.create_text(x_offset - 20, y_offset - 20, 
                               text="RED", fill="#FF6B6B", 
                               font=("Arial", 12, "bold"))
        
        # Bottom border (Red)
        bottom_y = y_offset + (board_size - 1) * y_spacing
        self.canvas.create_text(x_offset + (board_size - 1) * (x_spacing / 2) + 20, 
                               bottom_y + 20, 
                               text="RED", fill="#FF6B6B", 
                               font=("Arial", 12, "bold"))
        
        # Left border (Blue)
        self.canvas.create_text(x_offset - 30, 
                               y_offset + (board_size / 2) * y_spacing, 
                               text="BLUE", fill="#4ECDC4", 
                               font=("Arial", 12, "bold"), angle=90)
        
        # Right border (Blue)
        right_x = x_offset + (board_size - 1) * x_spacing + (board_size - 1) * (x_spacing / 2)
        self.canvas.create_text(right_x + 30, 
                               y_offset + (board_size / 2) * y_spacing, 
                               text="BLUE", fill="#4ECDC4", 
                               font=("Arial", 12, "bold"), angle=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HexGUI",
        description="Hex Game Visualizer - View games live or paste board states",
    )
    parser.add_argument(
        "-p1", "--player1",
        type=str,
        help="Player 1 agent (format: agents.GroupX.AgentFile AgentClassName)"
    )
    parser.add_argument(
        "-p2", "--player2",
        type=str,
        help="Player 2 agent (format: agents.GroupX.AgentFile AgentClassName)"
    )
    parser.add_argument(
        "-p1Name", "--player1Name",
        default="Alice",
        type=str,
        help="Player 1 name"
    )
    parser.add_argument(
        "-p2Name", "--player2Name",
        default="Bob",
        type=str,
        help="Player 2 name"
    )
    parser.add_argument(
        "-b", "--board_size",
        type=int,
        default=11,
        help="Board size"
    )
    
    args = parser.parse_args()
    
    root = tk.Tk()
    
    if args.player1 and args.player2:
        # Live game mode
        app = HexBoardGUI(
            root, 
            player1=args.player1,
            player2=args.player2,
            board_size=args.board_size,
            p1_name=args.player1Name,
            p2_name=args.player2Name
        )
    else:
        # Manual input mode
        app = HexBoardGUI(root)
    
    root.mainloop()
