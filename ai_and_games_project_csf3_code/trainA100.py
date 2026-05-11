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
GAMES_PER_ITERATION = 16384 # Reduced for faster feedback (approx 1h per iter)
MCTS_SIMS = 800            
EPOCHS = 10                
BATCH_SIZE = 16384         # Matches games count
LEARNING_RATE = 0.002
MODEL_FILE = "model_hpc.pt"

# Model Architecture
NUM_BLOCKS = 10            
NUM_FILTERS = 128          

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Training on: {device}")
if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")

# A100 Optimizations
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
    logging.info("Enabled TF32 precision for A100")
except AttributeError:
    pass

# ==========================================
# 2. C++ LIBRARY SETUP
# ==========================================
lib_path = os.path.join(os.path.dirname(__file__), "src/native/libmcts_hex.so")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Could not find C++ library at {lib_path}. Did you run compile_mcts.sh?")

lib = ctypes.CDLL(lib_path)

# Define argument types
lib.init.argtypes = [ctypes.c_int]
lib.reset_games.argtypes = [ctypes.c_int]
lib.mcts_prepare_batch_v2.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # board_tensor
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')    # indices
]
lib.mcts_update.argtypes = [
    ctypes.c_int,                                                   # count
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),   # indices
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # policies
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')  # values
]
lib.add_dirichlet_noise_all.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.play_moves.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # history_boards
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # history_probs
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
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                )
                for _ in range(num_blocks)
            ]
        )
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
# 4. SELF PLAY LOOP (C++ DRIVER)
# ==========================================
def generate_examples(model, num_games=GAMES_PER_ITERATION):
    # Initialize C++ games
    lib.reset_games(num_games)
    
    # Buffers for MCTS batching
    # Max batch size is num_games
    # Use pinned memory for faster CPU->GPU transfer
    board_tensor_np = torch.zeros((num_games, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, pin_memory=True).numpy()
    indices_np = np.zeros(num_games, dtype=np.int32)
    
    # Buffers for collecting training data
    # We assume max game length ~121. 
    # Buffer size: num_games * 121 is safe upper bound for one batch of finished games?
    # No, play_moves returns finished games.
    # We need a buffer large enough to hold the history of ALL games that might finish in one step.
    # In the worst case, all games finish at once.
    # Max history per game is 121.
    # So buffer size = num_games * 121.
    max_examples = num_games * 122
    hist_boards = np.zeros((max_examples, 3 * BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    hist_probs = np.zeros((max_examples, BOARD_SIZE * BOARD_SIZE + 1), dtype=np.float32)
    hist_vals = np.zeros(max_examples, dtype=np.float32)
    
    finished_games_total = 0
    all_examples = []
    
    # Prepare model for inference
    # Note: We avoid DataParallel here if using torch.compile()
    # But since we are not using torch.compile() anymore, we can keep DataParallel.
    # However, if we WANT to use torch.compile(), we must remove DataParallel.
    # DataParallel + compile = crashes.
    
    if torch.cuda.device_count() > 1:
        # Use DataParallel for 2x A100s (Better for 800 sims than 1x compiled)
        inference_model = nn.DataParallel(model)
    else:
        inference_model = model
    inference_model.eval()
    
    start_time = time.time()
    moves_total = 0
    
    logging.info(f"Starting self-play with {num_games} parallel games...")
    
    # We run until we have collected enough examples
    # Target: roughly num_games * 60 examples
    target_examples = num_games * 60
    
    while True:
        # 1. MCTS Simulations
        for i in range(MCTS_SIMS):
            # Get batch from C++
            count = lib.mcts_prepare_batch_v2(board_tensor_np, indices_np)
            
            if count > 0:
                # Slice the valid part
                # non_blocking=True works because source is pinned
                input_tensor = torch.from_numpy(board_tensor_np[:count]).to(device, memory_format=torch.channels_last, non_blocking=True)
                
                with torch.no_grad(), autocast('cuda'):
                    pis, vs = inference_model(input_tensor)
                
                pis_np = torch.softmax(pis, dim=1).cpu().numpy().astype(np.float32)
                vs_np = vs.cpu().numpy().flatten().astype(np.float32)
                
                # Update C++
                lib.mcts_update(count, indices_np, pis_np, vs_np)

            # Add Dirichlet Noise AFTER the first expansion (i=0)
            # This ensures the root has children to add noise to.
            if i == 0:
                lib.add_dirichlet_noise_all(num_games, 0.3, 0.25)
        
        # 2. Play Moves
        finished_count = ctypes.c_int(0)
        lib.play_moves(hist_boards, hist_probs, hist_vals, ctypes.byref(finished_count))
        
        moves_total += 1
        
        if finished_count.value > 0:
            n = finished_count.value
            
            # Extract data
            b_batch = hist_boards[:n].reshape(n, 3, BOARD_SIZE, BOARD_SIZE)
            p_batch = hist_probs[:n]
            v_batch = hist_vals[:n]
            
            all_examples.append((b_batch.copy(), p_batch.copy(), v_batch.copy()))
            
            current_examples = sum(len(x[0]) for x in all_examples)
            if current_examples >= target_examples:
                break
                
            print(f"Collected {current_examples}/{target_examples} examples... (Speed: {current_examples/(time.time()-start_time):.1f} ex/s)", end='\r')

    logging.info(f"Finished. Total examples: {sum(len(x[0]) for x in all_examples)}")
    
    # Merge all batches
    if not all_examples: return [], [], []
    
    final_b = np.concatenate([x[0] for x in all_examples])
    final_p = np.concatenate([x[1] for x in all_examples])
    final_v = np.concatenate([x[2] for x in all_examples])
    
    # Data Augmentation: 180-degree rotation
    # Hex board 180 rotation: (r, c) -> (10-r, 10-c)
    # This is equivalent to flipping both axes (np.flip)
    logging.info("Applying 180-degree rotation augmentation...")
    
    # Rotate boards (N, 3, 11, 11)
    # Flip last two dimensions (H, W)
    aug_b = np.flip(final_b, axis=(2, 3)).copy()
    
    # Rotate policies (N, 122)
    # The first 121 are board positions. The last one is SWAP.
    # Board positions map i -> 120 - i
    # Swap (index 121) stays at 121
    aug_p = np.zeros_like(final_p)
    aug_p[:, :121] = np.flip(final_p[:, :121], axis=1)
    aug_p[:, 121] = final_p[:, 121] # Swap move is invariant
    
    # Values are invariant
    aug_v = final_v.copy()
    
    # Concatenate original + augmented
    final_b = np.concatenate([final_b, aug_b])
    final_p = np.concatenate([final_p, aug_p])
    final_v = np.concatenate([final_v, aug_v])
    
    logging.info(f"Total examples after augmentation: {len(final_b)}")
    
    del all_examples
    gc.collect()

    return final_b, final_p, final_v

# ==========================================
# 5. TRAINING LOOP
# ==========================================
class HexDataset(Dataset):
    def __init__(self, boards, probs, vals):
        self.boards = boards
        self.probs = probs
        self.vals = vals
        
    def __len__(self):
        return len(self.boards)
        
    def __getitem__(self, idx):
        return self.boards[idx], self.probs[idx], self.vals[idx]

def train(model, dataset):
    if torch.cuda.device_count() > 1:
        train_model = nn.DataParallel(model)
    else:
        train_model = model

    optimizer = optim.Adam(train_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    scaler = GradScaler('cuda')

    train_model.train()
    total_loss = 0
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_v_loss = 0
        epoch_pi_loss = 0
        
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
            
            epoch_loss += loss.item()
            epoch_v_loss += loss_v.item()
            epoch_pi_loss += loss_pi.item()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_v = epoch_v_loss / len(dataloader)
        avg_pi = epoch_pi_loss / len(dataloader)
        
        logging.info(f"  Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f} (Val: {avg_v:.4f}, Pol: {avg_pi:.4f})")
        total_loss += epoch_loss
        
    scheduler.step(total_loss / (EPOCHS * len(dataloader)))

    gc.collect()
    return total_loss / (EPOCHS * len(dataloader))

def main():
    # Initialize C++ RNG
    lib.init(int(time.time()))
    
    model = HexResNet().to(device)
    model = model.to(memory_format=torch.channels_last) # Optimize for Tensor Cores
    
    if os.path.exists(MODEL_FILE):
        logging.info("Loading existing model...")
        state_dict = torch.load(MODEL_FILE, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle DataParallel prefix
            if k.startswith("module."):
                k = k[7:]
            # Handle torch.compile prefix
            if k.startswith("_orig_mod."):
                k = k[10:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # Compile model for faster inference (PyTorch 2.0+)
    # We re-enable this, but we MUST ensure DataParallel is NOT used in generate_examples
    # try:
    #     model = torch.compile(model)
    #     logging.info("Compiled model with torch.compile()")
    # except Exception as e:
    #     logging.warning(f"Could not compile model: {e}")

    for i in range(NUM_ITERATIONS):
        logging.info(f"=== Iteration {i+1}/{NUM_ITERATIONS} ===")
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if "VmRSS" in line:
                        logging.info(f"Memory Usage: {line.strip()}")
                        break
        except:
            pass

        # 1. Self Play
        boards, probs, vals = generate_examples(model, num_games=GAMES_PER_ITERATION)
        
        # 2. Train
        train_dataset = HexDataset(boards, probs, vals)
        loss = train(model, train_dataset)
        
        # 3. Save
        torch.save(model.state_dict(), MODEL_FILE)
        logging.info(f"Saved {MODEL_FILE}")

        # Check for golden model
        golden_filename = None
        if loss < 2.57:
            timestamp = int(time.time())
            golden_filename = f"model_hpc_golden_{loss:.4f}_{timestamp}.pt"
            torch.save(model.state_dict(), golden_filename)
            logging.info(f"Saved golden model {golden_filename} with loss {loss:.4f}")

        # 4. Sync
        save_dir = os.environ.get("SAVE_DIR")
        if save_dir:
            try:
                destination = os.path.join(save_dir, MODEL_FILE)
                shutil.copy2(MODEL_FILE, destination)
                logging.info(f"Synced model to {destination}")
                if golden_filename:
                    golden_destination = os.path.join(save_dir, golden_filename)
                    shutil.copy2(golden_filename, golden_destination)
                    logging.info(f"Synced golden model to {golden_destination}")
            except Exception as e:
                logging.warning(f"Failed to sync model: {e}")

if __name__ == "__main__":
    main()