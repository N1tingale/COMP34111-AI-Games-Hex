import os
import time
import math
import shutil
import ctypes
import gc
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ==========================================
# 1. HPC HYPERPARAMETERS
# ==========================================
BOARD_SIZE = 11
NUM_ITERATIONS = 100       
GAMES_PER_ITERATION = 16384 
MCTS_SIMS = 800            
EPOCHS = 10                
BATCH_SIZE = 4096 * 4      # Large batch for A100
LEARNING_RATE = 0.002
MODEL_FILE = "model_hpc.pt"

# AVX-512 Padding Constants
PADDED_SIZE = 128
ACTIONS_SIZE = 128

# Model Architecture
NUM_BLOCKS = 10            
NUM_FILTERS = 128          

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Training on: {device}")

# A100 Optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ==========================================
# 2. C++ LIBRARY SETUP
# ==========================================
lib_path = os.path.join(os.path.dirname(__file__), "libmcts_hex_optimised.so")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Could not find {lib_path}. Run compile_mcts_optimised.sh")

lib = ctypes.CDLL(lib_path)

# Argument types
lib.init.argtypes = [ctypes.c_int]
lib.reset_games.argtypes = [ctypes.c_int]
lib.mcts_prepare_batch_v2.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # board_tensor (Padded)
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')    # indices
]
lib.mcts_update.argtypes = [
    ctypes.c_int,                                                   # count
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # indices
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # policies (Padded)
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')  # values
]
lib.add_dirichlet_noise_all.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.play_moves.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # history_boards (Padded)
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # history_probs (Padded)
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # history_values
    ctypes.POINTER(ctypes.c_int)                                    # finished_count
]

# ==========================================
# 3. NEURAL NETWORK
# ==========================================
class HexResNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, num_blocks=NUM_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()
        self.board_size = board_size
        
        # Input is standard 11x11
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
            ) for _ in range(num_blocks)
        ])
        
        # Policy Head outputs 122 (11x11 + Swap)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1),
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv_input(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.relu(out + residual)
        return self.policy_head(out), self.value_head(out)

# ==========================================
# 4. SELF PLAY LOOP
# ==========================================
def generate_examples(model, num_games=GAMES_PER_ITERATION):
    lib.reset_games(num_games)
    
    # C++ communicates in PADDED TENSORS (128)
    # Shape: (N, 3, 128) -> Flattened: (N * 3 * 128)
    board_buffer_cpp = torch.zeros((num_games, 3, PADDED_SIZE), dtype=torch.float32, pin_memory=True).numpy()
    indices_np = np.zeros(num_games, dtype=np.int32)
    
    # History buffers
    max_examples = num_games * 125
    hist_boards = np.zeros((max_examples, 3 * PADDED_SIZE), dtype=np.float32)
    hist_probs = np.zeros((max_examples, ACTIONS_SIZE), dtype=np.float32)
    hist_vals = np.zeros(max_examples, dtype=np.float32)
    
    finished_games_total = 0
    all_examples = []
    
    # Use DataParallel only if not compiling, or handle compilation carefully
    # ideally for A100 80GB, single GPU batch inference is huge anyway.
    if torch.cuda.device_count() > 1:
        inference_model = nn.DataParallel(model)
    else:
        inference_model = model
    inference_model.eval()
    
    start_time = time.time()
    target_examples = num_games * 60
    
    while True:
        # 1. MCTS Simulations
        for i in range(MCTS_SIMS):
            count = lib.mcts_prepare_batch_v2(board_buffer_cpp, indices_np)
            
            if count > 0:
                # 1. Get Padded Tensor from C++
                raw_input = torch.from_numpy(board_buffer_cpp[:count]) # (N, 3, 128)
                
                # 2. Slice to Valid Board Size (11x11 = 121)
                # We assume C++ wrote planes contiguously.
                # Valid data is indices 0..120. Indices 121..127 are garbage padding.
                valid_input = raw_input[:, :, :BOARD_SIZE*BOARD_SIZE] # (N, 3, 121)
                
                # 3. Reshape to Geometry for Conv2d
                network_input = valid_input.view(count, 3, BOARD_SIZE, BOARD_SIZE).to(device, memory_format=torch.channels_last, non_blocking=True)
                
                with torch.no_grad(), autocast('cuda'):
                    pis, vs = inference_model(network_input)
                
                # 4. Pad Policy Output back to 128 for C++
                # pis shape: (N, 122). Target: (N, 128)
                pis_padded = F.pad(pis, (0, ACTIONS_SIZE - (BOARD_SIZE*BOARD_SIZE + 1)), "constant", 0)
                
                pis_np = torch.softmax(pis_padded, dim=1).cpu().numpy().astype(np.float32)
                vs_np = vs.cpu().numpy().flatten().astype(np.float32)
                
                lib.mcts_update(count, indices_np, pis_np, vs_np)

            if i == 0:
                lib.add_dirichlet_noise_all(num_games, 0.3, 0.25)
        
        # 2. Play Moves
        finished_count = ctypes.c_int(0)
        lib.play_moves(hist_boards, hist_probs, hist_vals, ctypes.byref(finished_count))
        
        if finished_count.value > 0:
            n = finished_count.value
            
            # Extract and Clean Data for Training
            # Raw: (N, 3*128)
            # We must UNPAD here to save disk space and for standard augmentation
            
            # Reshape to (N, 3, 128)
            raw_b = hist_boards[:n].reshape(n, 3, PADDED_SIZE)
            # Slice to 121
            clean_b = raw_b[:, :, :BOARD_SIZE*BOARD_SIZE]
            # Reshape to 11x11
            final_b = clean_b.reshape(n, 3, BOARD_SIZE, BOARD_SIZE).copy()
            
            # Policies: (N, 128) -> Slice to 122
            raw_p = hist_probs[:n]
            final_p = raw_p[:, :BOARD_SIZE*BOARD_SIZE + 1].copy()
            
            final_v = hist_vals[:n].copy()
            
            all_examples.append((final_b, final_p, final_v))
            
            curr_len = sum(len(x[0]) for x in all_examples)
            print(f"Collected {curr_len}/{target_examples}...", end='\r')
            if curr_len >= target_examples:
                break

    # Merge
    final_b = np.concatenate([x[0] for x in all_examples])
    final_p = np.concatenate([x[1] for x in all_examples])
    final_v = np.concatenate([x[2] for x in all_examples])
    
    # Augmentation (Standard 11x11 Logic now works!)
    aug_b = np.flip(final_b, axis=(2, 3)).copy()
    aug_p = np.zeros_like(final_p)
    aug_p[:, :121] = np.flip(final_p[:, :121], axis=1)
    aug_p[:, 121] = final_p[:, 121]
    aug_v = final_v.copy()
    
    final_b = np.concatenate([final_b, aug_b])
    final_p = np.concatenate([final_p, aug_p])
    final_v = np.concatenate([final_v, aug_v])
    
    return final_b, final_p, final_v

# ==========================================
# 5. TRAINING LOOP
# ==========================================
class HexDataset(Dataset):
    def __init__(self, boards, probs, vals):
        self.boards = boards
        self.probs = probs
        self.vals = vals
    def __len__(self): return len(self.boards)
    def __getitem__(self, idx): return self.boards[idx], self.probs[idx], self.vals[idx]

def train(model, dataset):
    # DDP/DataParallel
    train_model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    
    optimizer = optim.Adam(train_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    scaler = GradScaler('cuda')

    train_model.train()
    
    for epoch in range(EPOCHS):
        for boards, pis, vs in dataloader:
            boards = boards.to(device, memory_format=torch.channels_last)
            pis, vs = pis.to(device), vs.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                out_pi, out_v = train_model(boards)
                loss_v = F.mse_loss(out_v.view(-1), vs.float())
                loss_pi = -torch.sum(pis * F.log_softmax(out_pi, dim=1), dim=1).mean()
                loss = loss_v + loss_pi

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def main():
    lib.init(int(time.time()))
    model = HexResNet().to(device, memory_format=torch.channels_last)
    
    if os.path.exists(MODEL_FILE):
        logging.info("Loading model...")
        state_dict = torch.load(MODEL_FILE, map_location="cpu")
        # Clean keys
        new_state = {}
        for k,v in state_dict.items():
            k = k.replace("module.", "").replace("_orig_mod.", "")
            new_state[k] = v
        model.load_state_dict(new_state)

    for i in range(NUM_ITERATIONS):
        logging.info(f"Iteration {i+1}")
        boards, probs, vals = generate_examples(model)
        train_dataset = HexDataset(boards, probs, vals)
        train(model, train_dataset)
        torch.save(model.state_dict(), MODEL_FILE)

if __name__ == "__main__":
    main()