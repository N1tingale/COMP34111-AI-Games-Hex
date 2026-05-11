#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <deque>
#include <cstdint>
#include <immintrin.h> // AVX512
#include <omp.h>

#ifdef __linux__
#include <malloc.h>
#endif

// ==========================================
// CONSTANTS & MACROS
// ==========================================

#define EXPORT_API extern "C" 

const int BOARD_SIZE = 11;
const int NUM_TILES = BOARD_SIZE * BOARD_SIZE; // 121
const int ACTION_SWAP = NUM_TILES;             // 121
const int NUM_ACTIONS = NUM_TILES + 1;         // 122

// Padded sizes for AVX-512 (multiple of 16/32/64)
// We use 128 as it covers 122 and is divisible by 16 (for 512-bit floats)
const int NUM_TILES_SIZE = 128;
const int NUM_ACTIONS_SIZE = 128; 

// Colors
const int EMPTY = 0;
const int RED = 1;   // Vertical (Top-Bottom)
const int BLUE = 2;  // Horizontal (Left-Right)

// ==========================================
// DATA STRUCTURES
// ==========================================

// Union-Find for fast connectivity checks
struct UnionFind {
    std::vector<int> parent;
    
    void init(int n) {
        if (parent.size() < n) parent.resize(n);
        for(int i=0; i<n; ++i) parent[i] = i;
    }
    
    int find(int i) {
        if (parent[i] == i) return i;
        return parent[i] = find(parent[i]);
    }
    
    void unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) parent[root_i] = root_j;
    }
};

struct HexBoard {
    // Padded grids to prevent AVX out-of-bounds reads
    // 0 = empty, 1 = occupied (for specific color grids)
    // We use int8_t for compact storage, but AVX loads will handle converting to float masks
    alignas(64) int8_t grid[NUM_TILES_SIZE];      // 0=Empty, 1=Red, 2=Blue (Visualization only)
    alignas(64) int8_t grid_red[NUM_TILES_SIZE];  // 1 if Red, 0 else
    alignas(64) int8_t grid_blue[NUM_TILES_SIZE]; // 1 if Blue, 0 else
    alignas(64) int8_t grid_empty[NUM_TILES_SIZE];// 1 if Empty, 0 else
    
    int current_player; 
    int move_count;
    
    // Connectivity
    UnionFind group_red;
    UnionFind group_blue;
    
    HexBoard() {
        reset();
    }
    
    void reset();
    void update_groups(UnionFind& groups, int x, int y, int color);
    int check_winner();
    void get_valid_moves(std::vector<int>& moves);
    void apply_move(int action);
};

// MCTS Node
struct Node {
    int visit_count;
    float value_sum;
    float prior;
    int action;
    
    std::vector<Node*> children;
    
    Node(float p, int a) : visit_count(0), value_sum(0.0f), prior(p), action(a) {}
};

// Node Pool
struct NodePool {
    std::deque<Node> pool;
    size_t cursor;
    
    NodePool(size_t size) : cursor(0) {}
    
    void reset() {
        pool.clear();
        cursor = 0;
    }
    
    Node* alloc(float p, int a) {
        pool.emplace_back(p, a);
        return &pool.back();
    }
};

struct Game {
    HexBoard board;
    Node* root;
    NodePool pool;
    
    // Flattened history buffers
    // Note: Python expects these to be flat. 
    // We will store them in the padded format (128) for efficiency
    // and let Python handle the slicing.
    std::vector<float> history_planes_flat; 
    std::vector<float> history_probs_flat;  
    std::vector<int> history_players;
    
    Game() : pool(0) {
        root = pool.alloc(0.0f, -1);
    }
    
    ~Game() {} // Pool handles nodes
    
    void reset() {
        board.reset();
        pool.reset();
        root = pool.alloc(0.0f, -1);
        history_planes_flat.clear();
        history_probs_flat.clear();
        history_players.clear();
    }
};

// ==========================================
// IMPLEMENTATION
// ==========================================

// Fast Square Root approximation (Heron's Method)
inline float fast_sqrt(float n) {
    if (n <= 0) return 0;
    float x = n;
    // 2 iterations are usually enough for MCTS UCT
    x = 0.5f * (x + n / x);
    x = 0.5f * (x + n / x);
    return x;
}

void HexBoard::reset() {
    // Reset grids with padding safety
    std::memset(grid, 0, sizeof(grid));
    std::memset(grid_red, 0, sizeof(grid_red));
    std::memset(grid_blue, 0, sizeof(grid_blue));
    std::memset(grid_empty, 0, sizeof(grid_empty));
    
    // Mark valid board area as empty
    for(int i=0; i<NUM_TILES; ++i) {
        grid_empty[i] = 1;
    }
    // Padding area (121 to 127) stays 0 (occupied/invalid) so we never pick them

    current_player = RED;
    move_count = 0;

    // Connect borders to virtual nodes
    // Red: Top=NUM_TILES, Bottom=NUM_TILES+1
    // Blue: Left=NUM_TILES, Right=NUM_TILES+1
    group_red.init(NUM_TILES + 2);
    group_blue.init(NUM_TILES + 2);
}

void HexBoard::update_groups(UnionFind& groups, int x, int y, int color) {
    int idx = y * BOARD_SIZE + x;
    const int8_t* my_grid = (color == RED) ? grid_red : grid_blue;

    // Neighbors: (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x-1, y), (x-1, y+1)
    // Check bounds carefully
    
    // (x-1, y)
    if (x > 0 && my_grid[idx - 1]) groups.unite(idx, idx - 1);
    
    // (x-1, y+1)
    if (x > 0 && y < BOARD_SIZE - 1 && my_grid[idx + BOARD_SIZE - 1]) groups.unite(idx, idx + BOARD_SIZE - 1);
    
    // (x+1, y)
    if (x < BOARD_SIZE - 1 && my_grid[idx + 1]) groups.unite(idx, idx + 1);
    
    // (x+1, y-1)
    if (x < BOARD_SIZE - 1 && y > 0 && my_grid[idx - BOARD_SIZE + 1]) groups.unite(idx, idx - BOARD_SIZE + 1);
    
    // (x, y-1)
    if (y > 0 && my_grid[idx - BOARD_SIZE]) groups.unite(idx, idx - BOARD_SIZE);
    
    // (x, y+1)
    if (y < BOARD_SIZE - 1 && my_grid[idx + BOARD_SIZE]) groups.unite(idx, idx + BOARD_SIZE);
}

int HexBoard::check_winner() {
    if (group_red.find(NUM_TILES) == group_red.find(NUM_TILES + 1)) return RED;
    if (group_blue.find(NUM_TILES) == group_blue.find(NUM_TILES + 1)) return BLUE;
    return 0;
}

void HexBoard::get_valid_moves(std::vector<int>& moves) {
    moves.clear();
    // Only iterate up to valid board size
    for (int i = 0; i < NUM_TILES; ++i) {
        if (grid_empty[i]) moves.push_back(i);
    }
    if (move_count == 1) moves.push_back(ACTION_SWAP);
}

void HexBoard::apply_move(int action) {
    if (action == ACTION_SWAP) {
        // Swap implementation
        int p1_move = -1;
        for(int i=0; i<NUM_TILES; ++i) {
            if (!grid_empty[i]) {
                p1_move = i;
                break;
            }
        }
        
        if (p1_move != -1) {
            // Remove old stone
            grid_empty[p1_move] = 1;
            grid_red[p1_move] = 0;
            grid_blue[p1_move] = 0;
            grid[p1_move] = 0;
            
            group_red.init(NUM_TILES + 2);
            group_blue.init(NUM_TILES + 2);
            
            // Transpose
            int ox = p1_move % BOARD_SIZE;
            int oy = p1_move / BOARD_SIZE;
            int new_move = ox * BOARD_SIZE + oy;
            
            // Place as current player
            grid_empty[new_move] = 0;
            if (current_player == RED) grid_red[new_move] = 1;
            else grid_blue[new_move] = 1;
            grid[new_move] = current_player;
            
            // Update UF
            int nx = new_move % BOARD_SIZE;
            int ny = new_move / BOARD_SIZE;
            
            if (current_player == RED) {
                if (ny == 0) group_red.unite(new_move, NUM_TILES);
                if (ny == BOARD_SIZE - 1) group_red.unite(new_move, NUM_TILES + 1);
            } else {
                if (nx == 0) group_blue.unite(new_move, NUM_TILES);
                if (nx == BOARD_SIZE - 1) group_blue.unite(new_move, NUM_TILES + 1);
            }
        }
        
        current_player = (current_player == RED) ? BLUE : RED;
        move_count++;
        return;
    }
    
    // Normal move
    int x = action % BOARD_SIZE;
    int y = action / BOARD_SIZE;
    
    grid_empty[action] = 0;
    grid[action] = current_player;
    
    if (current_player == RED) {
        grid_red[action] = 1;
        update_groups(group_red, x, y, RED);
        if (y == 0) group_red.unite(action, NUM_TILES);
        if (y == BOARD_SIZE - 1) group_red.unite(action, NUM_TILES + 1);
    } else {
        grid_blue[action] = 1;
        update_groups(group_blue, x, y, BLUE);
        if (x == 0) group_blue.unite(action, NUM_TILES);
        if (x == BOARD_SIZE - 1) group_blue.unite(action, NUM_TILES + 1);
    }
    
    current_player = (current_player == RED) ? BLUE : RED;
    move_count++;
}

// Global State
std::vector<Game*> games;
std::vector<std::vector<Node*>> stored_paths;
int global_seed = 42;
#pragma omp threadprivate(global_seed)

// ==========================================
// EXPORTED FUNCTIONS
// ==========================================

EXPORT_API void init(int seed) {
    global_seed = seed;
}

EXPORT_API void reset_games(int count) {
    for (Game* g : games) delete g;
    games.clear();
    #ifdef __linux__
    malloc_trim(0);
    #endif
    for (int i = 0; i < count; ++i) {
        games.push_back(new Game());
    }
}

EXPORT_API void add_dirichlet_noise(int game_idx, float alpha, float epsilon) {
    if (game_idx < 0 || game_idx >= (int)games.size()) return;
    Game* g = games[game_idx];
    if (!g->root || g->root->children.empty()) return;

    std::mt19937 local_rng(global_seed + game_idx);
    std::gamma_distribution<float> distribution(alpha, 1.0f);
    float sum = 0.0f;
    std::vector<float> noise(g->root->children.size());

    for (size_t i = 0; i < g->root->children.size(); ++i) {
        noise[i] = distribution(local_rng);
        sum += noise[i];
    }

    for (size_t i = 0; i < g->root->children.size(); ++i) {
        Node* child = g->root->children[i];
        child->prior = (1 - epsilon) * child->prior + epsilon * (noise[i] / sum);
    }
}

EXPORT_API void add_dirichlet_noise_all(int count, float alpha, float epsilon) {
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        add_dirichlet_noise(i, alpha, epsilon);
    }
}

// Prepare MCTS Batch (Vectorized)
// Returns number of active leaf nodes found
EXPORT_API int mcts_prepare_batch_v2(float* board_tensor, int* indices) {
    stored_paths.clear();
    stored_paths.resize(games.size());
    
    std::vector<int> active_games;
    active_games.reserve(games.size());
    for(int i=0; i<(int)games.size(); ++i) {
        if (games[i]->board.check_winner() == 0) {
            active_games.push_back(i);
        }
    }
    
    if (active_games.empty()) return 0;
    
    static std::vector<int> batch_indices; // Static to reduce realloc
    batch_indices.clear();
    
    // Thread-local buffers to minimize critical sections
    #pragma omp parallel
    {
        std::vector<int> local_batch_indices;
        std::vector<float> local_tensor;
        
        #pragma omp for
        for (int k = 0; k < (int)active_games.size(); ++k) {
            int idx = active_games[k];
            Game* g = games[idx];
            Node* node = g->root;
            std::vector<Node*>& path = stored_paths[idx];
            
            HexBoard sim_board;
            // Using default copy assignment which memcpy's the arrays
            sim_board = g->board; 
            
            path.push_back(node);
            
            // Selection
            while (!node->children.empty() && sim_board.check_winner() == 0) {
                float best_score = -1e9;
                Node* best_child = nullptr;
                int best_action = -1;
                float sqrt_visit = fast_sqrt((float)node->visit_count);
                
                for (Node* child : node->children) {
                    float q = (child->visit_count > 0) ? -child->value_sum / child->visit_count : 0.0f;
                    float u = 1.4f * child->prior * sqrt_visit / (1.0f + child->visit_count);
                    if (q + u > best_score) {
                        best_score = q + u;
                        best_child = child;
                        best_action = child->action;
                    }
                }
                
                if (best_child) {
                    node = best_child;
                    sim_board.apply_move(best_action);
                    path.push_back(node);
                } else break;
            }
            
            int winner = sim_board.check_winner();
            if (winner != 0) {
                // Terminal handling (Backprop immediately)
                int p = g->board.current_player;
                for (size_t j=0; j<path.size(); ++j) {
                    path[j]->visit_count++;
                    float v = (winner == p) ? 1.0f : -1.0f;
                    path[j]->value_sum += v;
                    
                    if (j+1 < path.size()) p = (p==RED) ? BLUE : RED;
                }
            } else {
                // Add to batch for NN evaluation
                local_batch_indices.push_back(idx);
                
                int my = sim_board.current_player;
                int opp = (my == RED) ? BLUE : RED;
                
                // --- AVX-512 Tensor Construction ---
                // We write 3 planes of 128 floats each
                int pos = local_tensor.size();
                local_tensor.resize(pos + 3 * NUM_TILES_SIZE);
                
                const __m512 f_zero = _mm512_setzero_ps();
                const __m512 f_one = _mm512_set1_ps(1.0f);
                
                // Plane 0: My Stones
                const int8_t* my_grid = (my == RED) ? sim_board.grid_red : sim_board.grid_blue;
                for (int i=0; i < NUM_TILES_SIZE/16; ++i) {
                    // Load 16 bytes (int8)
                    // We need to extend this to mask.
                    // Load 128 bits -> convert to mask? 
                    // Simpler: Load 16 chars, create mask where val == 1
                    
                    __m128i chars = _mm_load_si128((__m128i*)&my_grid[i*16]);
                    __mmask16 mask = _mm_test_epi8_mask(chars, chars); // Non-zero implies set
                    
                    __m512 res = _mm512_mask_blend_ps(mask, f_zero, f_one);
                    _mm512_storeu_ps(&local_tensor[pos + i*16], res);
                }
                pos += NUM_TILES_SIZE;
                
                // Plane 1: Opponent Stones
                const int8_t* opp_grid = (opp == RED) ? sim_board.grid_red : sim_board.grid_blue;
                for (int i=0; i < NUM_TILES_SIZE/16; ++i) {
                    __m128i chars = _mm_load_si128((__m128i*)&opp_grid[i*16]);
                    __mmask16 mask = _mm_test_epi8_mask(chars, chars);
                    
                    __m512 res = _mm512_mask_blend_ps(mask, f_zero, f_one);
                    _mm512_storeu_ps(&local_tensor[pos + i*16], res);
                }
                pos += NUM_TILES_SIZE;
                
                // Plane 2: Color (All 1.0 or All 0.0)
                __m512 color_val = (my == RED) ? f_one : f_zero;
                for (int i=0; i < NUM_TILES_SIZE/16; ++i) {
                    _mm512_storeu_ps(&local_tensor[pos + i*16], color_val);
                }
            }
        }
        
        #pragma omp critical
        {
            int offset = batch_indices.size();
            batch_indices.insert(batch_indices.end(), local_batch_indices.begin(), local_batch_indices.end());
            
            // Copy local tensor data to global pointer
            // local_tensor is flattened: [Game1_P0, Game1_P1, Game1_P2, Game2_P0...]
            for (size_t i=0; i<local_batch_indices.size(); ++i) {
                indices[offset + i] = local_batch_indices[i];
                int global_offset = (offset + i) * 3 * NUM_TILES_SIZE;
                int local_offset = i * 3 * NUM_TILES_SIZE;
                std::memcpy(&board_tensor[global_offset], &local_tensor[local_offset], 3 * NUM_TILES_SIZE * sizeof(float));
            }
        }
    }
    
    return batch_indices.size();
}

EXPORT_API void mcts_update(int count, int* indices, float* policies, float* values) {
    #pragma omp parallel for
    for (int i=0; i<count; ++i) {
        int idx = indices[i];
        float val = values[i];
        
        Game* g = games[idx];
        std::vector<Node*>& path = stored_paths[idx];
        Node* leaf = path.back();
        
        // Resimulate to get valid moves
        HexBoard sim_board = g->board; // Copy
        for (size_t k=1; k<path.size(); ++k) {
            sim_board.apply_move(path[k]->action);
        }
        
        std::vector<int> moves;
        sim_board.get_valid_moves(moves);
        
        // Sum policies for valid moves
        float policy_sum = 0.0f;
        // Policies come in as padded 128-float arrays
        int pol_offset = i * NUM_ACTIONS_SIZE; 
        
        for (int move : moves) {
            policy_sum += policies[pol_offset + move];
        }
        
        if (policy_sum > 1e-6) {
            for (int move : moves) {
                float p = policies[pol_offset + move] / policy_sum;
                leaf->children.push_back(g->pool.alloc(p, move));
            }
        } else {
            float p = 1.0f / moves.size();
            for (int move : moves) {
                leaf->children.push_back(g->pool.alloc(p, move));
            }
        }
        
        // Backprop
        for (int k=path.size()-1; k>=0; --k) {
            path[k]->visit_count++;
            path[k]->value_sum += val;
            val = -val;
        }
    }
}

EXPORT_API int play_moves(float* history_boards, float* history_probs, float* history_values, int* finished_count) {
    int num_games = games.size();
    int example_count = 0;
    
    // Using static thread_local RNG
    static thread_local std::mt19937 rng(global_seed + 1); // +1 to differ from init
    
    // We execute serially or parallel? Parallel is fine if we manage the output buffer atomically
    // But copying to output buffer is safer done serially or with calculated offsets.
    // Given play_moves is fast (just ArgMax), let's do parallel gathering then serial write.
    
    struct Result {
        std::vector<float> boards;
        std::vector<float> probs;
        std::vector<float> values;
        int count;
    };
    
    std::vector<Result> thread_results(omp_get_max_threads());
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Result& res = thread_results[tid];
        res.count = 0;
        
        #pragma omp for
        for (int i=0; i<num_games; ++i) {
            Game* g = games[i];
            if (g->board.check_winner() != 0) continue;
            
            Node* root = g->root;
            
            // 1. Calc Probs
            std::vector<float> probs(NUM_ACTIONS_SIZE, 0.0f);
            float sum_visits = 0;
            for (Node* child : root->children) {
                probs[child->action] = (float)child->visit_count;
                sum_visits += child->visit_count;
            }
            
            if (sum_visits == 0) {
                 // Random fallback
                 std::vector<int> moves;
                 g->board.get_valid_moves(moves);
                 int action = moves[rng() % moves.size()];
                 g->board.apply_move(action);
                 g->pool.reset();
                 g->root = g->pool.alloc(0, -1);
                 continue;
            }
            
            // Normalize (AVX)
            __m512 v_sum = _mm512_set1_ps(sum_visits);
            for (int k=0; k<NUM_ACTIONS_SIZE/16; ++k) {
                __m512 v_probs = _mm512_loadu_ps(&probs[k*16]);
                _mm512_storeu_ps(&probs[k*16], _mm512_div_ps(v_probs, v_sum));
            }
            
            // 2. Select Action
            int chosen_action = -1;
            if (g->board.move_count < 15) {
                // Softmax sampling / Visit Count sampling
                float r = std::uniform_real_distribution<float>(0,1)(rng);
                float cum = 0;
                for (int k=0; k<NUM_ACTIONS; ++k) {
                    cum += probs[k];
                    if (r <= cum) {
                        chosen_action = k;
                        break;
                    }
                }
                if (chosen_action == -1) chosen_action = NUM_ACTIONS-1; // Swap?
            } else {
                float max_p = -1;
                for (int k=0; k<NUM_ACTIONS; ++k) {
                    if (probs[k] > max_p) {
                        max_p = probs[k];
                        chosen_action = k;
                    }
                }
            }
            
            // 3. Store History
            // History planes are already in 128-padded format
            // We reuse the AVX generation logic from prepare_batch_v2?
            // Or just do it here.
            int my = g->board.current_player;
            int opp = (my==RED)?BLUE:RED;
            
            const __m512 f_zero = _mm512_setzero_ps();
            const __m512 f_one = _mm512_set1_ps(1.0f);
            
            // Append 3 * 128 floats
            int start_idx = g->history_planes_flat.size();
            g->history_planes_flat.resize(start_idx + 3 * NUM_TILES_SIZE);
            float* ptr = &g->history_planes_flat[start_idx];
            
            const int8_t* my_grid = (my==RED) ? g->board.grid_red : g->board.grid_blue;
            for(int k=0; k<NUM_TILES_SIZE/16; ++k) {
                 __mmask16 mask = _mm_test_epi8_mask(_mm_load_si128((__m128i*)&my_grid[k*16]), _mm_load_si128((__m128i*)&my_grid[k*16]));
                 _mm512_storeu_ps(ptr + k*16, _mm512_mask_blend_ps(mask, f_zero, f_one));
            }
            ptr += NUM_TILES_SIZE;
            
            const int8_t* opp_grid = (opp==RED) ? g->board.grid_red : g->board.grid_blue;
            for(int k=0; k<NUM_TILES_SIZE/16; ++k) {
                 __mmask16 mask = _mm_test_epi8_mask(_mm_load_si128((__m128i*)&opp_grid[k*16]), _mm_load_si128((__m128i*)&opp_grid[k*16]));
                 _mm512_storeu_ps(ptr + k*16, _mm512_mask_blend_ps(mask, f_zero, f_one));
            }
            ptr += NUM_TILES_SIZE;
            
            __m512 c_val = (my==RED) ? f_one : f_zero;
            for(int k=0; k<NUM_TILES_SIZE/16; ++k) {
                _mm512_storeu_ps(ptr + k*16, c_val);
            }
            
            // Store Probs
            g->history_probs_flat.insert(g->history_probs_flat.end(), probs.begin(), probs.end());
            g->history_players.push_back(my);
            
            // 4. Apply Move & Reset
            g->board.apply_move(chosen_action);
            g->pool.reset();
            g->root = g->pool.alloc(0, -1);
            
            // 5. Check Winner
            int winner = g->board.check_winner();
            if (winner != 0) {
                // Collect results into thread local buffer
                int moves_in_game = g->history_players.size();
                res.count += moves_in_game;
                
                for (int m=0; m<moves_in_game; ++m) {
                    int p = g->history_players[m];
                    float v = (p == winner) ? 1.0f : -1.0f;
                    
                    // Copy board (3*128)
                    int b_idx = m * 3 * NUM_TILES_SIZE;
                    for(int x=0; x<3*NUM_TILES_SIZE; ++x) res.boards.push_back(g->history_planes_flat[b_idx+x]);
                    
                    // Copy probs (128)
                    int p_idx = m * NUM_ACTIONS_SIZE;
                    for(int x=0; x<NUM_ACTIONS_SIZE; ++x) res.probs.push_back(g->history_probs_flat[p_idx+x]);
                    
                    res.values.push_back(v);
                }
                
                g->reset();
            }
        }
    }
    
    // Aggregate results
    int offset = 0;
    for (const auto& res : thread_results) {
        if (res.count == 0) continue;
        
        std::memcpy(history_boards + offset * 3 * NUM_TILES_SIZE, res.boards.data(), res.boards.size() * sizeof(float));
        std::memcpy(history_probs + offset * NUM_ACTIONS_SIZE, res.probs.data(), res.probs.size() * sizeof(float));
        std::memcpy(history_values + offset, res.values.data(), res.values.size() * sizeof(float));
        
        offset += res.count;
    }
    
    *finished_count = offset;
    return 0;
}