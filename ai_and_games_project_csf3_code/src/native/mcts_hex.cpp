#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <deque>
#include <cstdint>

#ifdef __linux__
#include <malloc.h>
#endif

// Constants
const int BOARD_SIZE = 11;
const int NUM_TILES = BOARD_SIZE * BOARD_SIZE;
const int ACTION_SWAP = NUM_TILES;
const int NUM_ACTIONS = NUM_TILES + 1;

// Colors
const int EMPTY = 0;
const int RED = 1;   // Vertical (Top-Bottom)
const int BLUE = 2;  // Horizontal (Left-Right)

// Union-Find for fast connectivity checks
struct UnionFind {
    std::vector<int> parent;
    
    void init(int n) {
        parent.resize(n);
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
    int8_t grid[NUM_TILES];
    int current_player; // 1 or 2
    int move_count;
    
    // Connectivity
    UnionFind uf_red;
    UnionFind uf_blue;
    
    // Virtual nodes for UF
    // Red: Top=NUM_TILES, Bottom=NUM_TILES+1
    // Blue: Left=NUM_TILES, Right=NUM_TILES+1
    
    HexBoard() {
        reset();
    }
    
    void reset() {
        std::memset(grid, 0, sizeof(grid));
        current_player = RED;
        move_count = 0;
        uf_red.init(NUM_TILES + 2);
        uf_blue.init(NUM_TILES + 2);
        
        // Connect borders to virtual nodes
        for (int i = 0; i < BOARD_SIZE; ++i) {
            // Red Top (Row 0) -> Node NUM_TILES
            // Red Bottom (Row 10) -> Node NUM_TILES+1
            // Blue Left (Col 0) -> Node NUM_TILES
            // Blue Right (Col 10) -> Node NUM_TILES+1
            // We handle this dynamically when stones are placed
        }
    }
    
    void copy_from(const HexBoard& other) {
        std::memcpy(grid, other.grid, sizeof(grid));
        current_player = other.current_player;
        move_count = other.move_count;
        uf_red = other.uf_red;
        uf_blue = other.uf_blue;
    }
    
    void apply_move(int action) {
        if (action == ACTION_SWAP) {
            // Swap Rule (Pie Rule):
            // The second player (Current Player) chooses to swap roles.
            // In implementation terms: We steal the first player's move.
            // We find the single stone on the board (placed by P1).
            // We move it to its transpose (r,c) -> (c,r).
            // We change its color to OUR color (Current Player).
            // Effectively, it's as if WE played that transposed move as our first move.
            
            int p1_move = -1;
            for(int i=0; i<NUM_TILES; ++i) {
                if (grid[i] != EMPTY) {
                    p1_move = i;
                    break;
                }
            }
            
            // Clear the old stone
            if (p1_move != -1) {
                grid[p1_move] = EMPTY;
                // We must rebuild UF because removing a stone is hard in UF
                // But since there was only 1 stone, we can just reset UF.
                uf_red.init(NUM_TILES + 2);
                uf_blue.init(NUM_TILES + 2);
                
                // Calculate transpose
                int r = p1_move / BOARD_SIZE;
                int c = p1_move % BOARD_SIZE;
                int new_move = c * BOARD_SIZE + r;
                
                // Place stone as Current Player
                grid[new_move] = current_player;
                
                // Update UF for the new single stone
                int nr = new_move / BOARD_SIZE;
                int nc = new_move % BOARD_SIZE;
                
                if (current_player == RED) {
                    if (nr == 0) uf_red.unite(new_move, NUM_TILES);
                    if (nr == BOARD_SIZE - 1) uf_red.unite(new_move, NUM_TILES + 1);
                } else {
                    if (nc == 0) uf_blue.unite(new_move, NUM_TILES);
                    if (nc == BOARD_SIZE - 1) uf_blue.unite(new_move, NUM_TILES + 1);
                }
            }
            
            current_player = (current_player == RED) ? BLUE : RED;
            move_count++;
            return;
        }
        
        int r = action / BOARD_SIZE;
        int c = action % BOARD_SIZE;
        
        grid[action] = current_player;
        
        // Update Union-Find
        if (current_player == RED) {
            // Connect to neighbors
            update_connectivity(uf_red, r, c, RED);
            // Connect to borders
            if (r == 0) uf_red.unite(action, NUM_TILES);
            if (r == BOARD_SIZE - 1) uf_red.unite(action, NUM_TILES + 1);
        } else {
            update_connectivity(uf_blue, r, c, BLUE);
            if (c == 0) uf_blue.unite(action, NUM_TILES);
            if (c == BOARD_SIZE - 1) uf_blue.unite(action, NUM_TILES + 1);
        }
        
        current_player = (current_player == RED) ? BLUE : RED;
        move_count++;
    }
    
    void update_connectivity(UnionFind& uf, int r, int c, int color) {
        int idx = r * BOARD_SIZE + c;
        // Neighbors: (r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c)
        int dr[] = {-1, -1, 0, 0, 1, 1};
        int dc[] = {0, 1, -1, 1, -1, 0};
        
        for (int i = 0; i < 6; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                int nidx = nr * BOARD_SIZE + nc;
                if (grid[nidx] == color) {
                    uf.unite(idx, nidx);
                }
            }
        }
    }
    
    int check_winner() {
        if (uf_red.find(NUM_TILES) == uf_red.find(NUM_TILES + 1)) return RED;
        if (uf_blue.find(NUM_TILES) == uf_blue.find(NUM_TILES + 1)) return BLUE;
        return 0;
    }
    
    void get_valid_moves(std::vector<int>& moves) {
        moves.clear();
        for (int i = 0; i < NUM_TILES; ++i) {
            if (grid[i] == EMPTY) moves.push_back(i);
        }
        if (move_count == 1) moves.push_back(ACTION_SWAP);
    }
};

// Forward declaration
struct Node;

// Node Pool to avoid fragmentation
struct NodePool {
    std::deque<Node> pool;
    size_t cursor;
    
    NodePool(size_t size);
    
    void reset();
    
    Node* alloc(float p, int a);
};

// MCTS Node
struct Node {
    int visit_count;
    float value_sum;
    float prior;
    int action;
    
    std::vector<Node*> children;
    
    Node(float p, int a) : visit_count(0), value_sum(0.0f), prior(p), action(a) {}
    
    // No destructor needed for children as they are in pool
};

// NodePool Implementation
NodePool::NodePool(size_t size) {
    // Deque doesn't have reserve, but we can't pre-allocate easily without default constructor
    // Actually we don't need to reserve with deque as it grows stably.
    cursor = 0;
}

void NodePool::reset() {
    pool.clear();
    cursor = 0;
}

Node* NodePool::alloc(float p, int a) {
    pool.emplace_back(p, a);
    return &pool.back();
}

// Game Wrapper
struct Game {
    HexBoard board;
    Node* root;
    NodePool pool;
    std::vector<float> history_planes_flat; // Flattened 3x11x11
    std::vector<float> history_probs_flat;  // Flattened
    std::vector<int> history_players;
    
    Game() : pool(0) { // Size ignored for deque
        root = pool.alloc(0.0f, -1);
    }
    
    ~Game() {
        // Pool cleans up nodes
    }
    
    void reset() {
        board.reset();
        pool.reset();
        root = pool.alloc(0.0f, -1);
        history_planes_flat.clear();
        history_probs_flat.clear();
        history_players.clear();
        history_planes_flat.shrink_to_fit();
        history_probs_flat.shrink_to_fit();
        history_players.shrink_to_fit();
    }
};

// Global State
std::vector<Game*> games;
int global_seed = 42;
#pragma omp threadprivate(global_seed)

extern "C" {

void init(int seed) {
    global_seed = seed;
}

void reset_games(int count) {
    for (Game* g : games) delete g;
    games.clear();
    #ifdef __linux__
    malloc_trim(0);
    #endif
    for (int i = 0; i < count; ++i) {
        games.push_back(new Game());
    }
}

void add_dirichlet_noise(int game_idx, float alpha, float epsilon) {
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
        float n = noise[i] / sum;
        child->prior = (1 - epsilon) * child->prior + epsilon * n;
    }
}

void add_dirichlet_noise_all(int count, float alpha, float epsilon) {
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        add_dirichlet_noise(i, alpha, epsilon);
    }
}

// Returns number of active games that need evaluation
// Fills board_tensor with (N, 3, 11, 11) float data
// Fills indices with the game index for each batch item
int mcts_prepare_batch(float* board_tensor, int* indices, int mcts_sims) {
    int num_games = games.size();
    
    // We can parallelize this loop, but we need to synchronize access to batch_count
    // For simplicity and safety with the shared buffer, we might do a two-pass or use critical section
    // But since we are just reading and writing to distinct parts of buffer, we can calculate offsets?
    // No, we don't know which games are active.
    
    // Let's just do it serially or with a thread-local buffer then merge.
    // Given 512 games, serial selection is fast enough in C++.
    
    // Wait, the request is to do MCTS steps.
    // We need to traverse the tree for EACH game until we hit a leaf.
    
    std::vector<int> active_games;
    for(int i=0; i<num_games; ++i) {
        if (games[i]->board.check_winner() == 0) {
            active_games.push_back(i);
        }
    }
    
    if (active_games.empty()) return 0;

    // Temporary storage for paths to avoid re-traversing
    // But we can't store paths easily across C/Python boundary without complex state.
    // We will just traverse, find leaf, write to tensor.
    // Python will call update, we will use the indices to find the node again?
    // No, that's inefficient (double traversal).
    // Better: Store the leaf node pointer in a temporary global vector.
    
    static std::vector<Node*> leaf_nodes;
    static std::vector<int> leaf_game_indices;
    static std::vector<int> leaf_players;
    
    leaf_nodes.clear();
    leaf_game_indices.clear();
    leaf_players.clear();
    
    // Parallel Traversal
    #pragma omp parallel
    {
        std::vector<Node*> local_nodes;
        std::vector<int> local_indices;
        std::vector<int> local_players;
        std::vector<float> local_tensor_data; // Flattened
        
        #pragma omp for
        for (int k = 0; k < (int)active_games.size(); ++k) {
            int idx = active_games[k];
            Game* g = games[idx];
            Node* node = g->root;
            HexBoard sim_board;
            sim_board.copy_from(g->board);
            
            // Select
            while (!node->children.empty() && sim_board.check_winner() == 0) {
                float best_score = -1e9;
                Node* best_child = nullptr;
                int best_action = -1;
                
                float sqrt_visit = std::sqrt((float)node->visit_count);
                
                for (Node* child : node->children) {
                    float q = 0;
                    if (child->visit_count > 0) {
                        q = -child->value_sum / child->visit_count; // Negate value for opponent
                    }
                    float u = 1.4f * child->prior * sqrt_visit / (1.0f + child->visit_count);
                    float score = q + u;
                    
                    if (score > best_score) {
                        best_score = score;
                        best_child = child;
                        best_action = child->action;
                    }
                }
                
                if (best_child) {
                    node = best_child;
                    sim_board.apply_move(best_action);
                } else {
                    break; // Should not happen
                }
            }
            
            // We are at a leaf.
            // Check if terminal
            int winner = sim_board.check_winner();
            if (winner != 0) {
                // Terminal node. Backprop immediately.
                // Value is 1 if winner == current_player, else -1
                // But wait, we need to backprop up the path.
                // We didn't store the path!
                // Re-traversal is needed or we need to store path in stack.
                // Storing path in stack is better.
                // Let's refactor to store path.
            } else {
                // Not terminal, need evaluation.
                local_nodes.push_back(node);
                local_indices.push_back(idx);
                local_players.push_back(sim_board.current_player);
                
                // Write to tensor buffer
                // 3 x 11 x 11
                // Plane 0: My stones
                // Plane 1: Opp stones
                // Plane 2: Turn (all 1)
                int my = sim_board.current_player;
                int opp = (my == RED) ? BLUE : RED;
                
                // We just push to a local vector and copy later to avoid false sharing
                // 3 * 121 = 363 floats
                for (int r=0; r<BOARD_SIZE; ++r) {
                    for (int c=0; c<BOARD_SIZE; ++c) {
                        int t = r*BOARD_SIZE + c;
                        int val = sim_board.grid[t];
                        local_tensor_data.push_back(val == my ? 1.0f : 0.0f);
                    }
                }
                for (int r=0; r<BOARD_SIZE; ++r) {
                    for (int c=0; c<BOARD_SIZE; ++c) {
                        int t = r*BOARD_SIZE + c;
                        int val = sim_board.grid[t];
                        local_tensor_data.push_back(val == opp ? 1.0f : 0.0f);
                    }
                }
                for (int i=0; i<NUM_TILES; ++i) local_tensor_data.push_back(1.0f);
            }
        }
        
        #pragma omp critical
        {
            int offset = leaf_nodes.size();
            leaf_nodes.insert(leaf_nodes.end(), local_nodes.begin(), local_nodes.end());
            leaf_game_indices.insert(leaf_game_indices.end(), local_indices.begin(), local_indices.end());
            leaf_players.insert(leaf_players.end(), local_players.begin(), local_players.end());
            
            // Copy tensor data
            // Each item is 363 floats
            for (int i=0; i<(int)local_nodes.size(); ++i) {
                int base = (offset + i) * 3 * NUM_TILES;
                int local_base = i * 3 * NUM_TILES;
                for (int j=0; j<3*NUM_TILES; ++j) {
                    board_tensor[base + j] = local_tensor_data[local_base + j];
                }
                indices[offset + i] = local_indices[i];
            }
        }
    }
    
    // Handle terminal nodes separately?
    // For simplicity in this "Lite" version, we will just let the Python loop handle the non-terminal ones.
    // Terminal ones are rare during selection until end of game.
    // Actually, if we hit a terminal node, we should backpropagate immediately.
    // But we lost the path.
    // To fix this properly, we need to store the path.
    // But we can't easily pass the path around.
    // Solution: We will just add the terminal nodes to the batch with a "dummy" evaluation
    // where value is correct and policy is 0.
    // But we need to know the winner.
    // Let's stick to the "Evaluation" phase.
    
    return leaf_nodes.size();
}

// Updates the leaf nodes with policy and value, expands them, and backpropagates
// BUT wait, we need to backpropagate up the tree.
// Since we didn't store the path in `prepare_batch`, we can't backpropagate!
// We MUST store the path.
// Since `prepare_batch` returns to Python, the stack is lost.
// We need to store the paths in a global `vector<vector<Node*>> paths`.

std::vector<std::vector<Node*>> stored_paths;

int mcts_prepare_batch_v2(float* board_tensor, int* indices) {
    stored_paths.clear();
    stored_paths.resize(games.size()); // We might not use all
    
    std::vector<int> active_games;
    for(int i=0; i<(int)games.size(); ++i) {
        if (games[i]->board.check_winner() == 0) {
            active_games.push_back(i);
        }
    }
    
    if (active_games.empty()) return 0;
    
    // Global vectors to collect batch
    static std::vector<int> batch_indices;
    batch_indices.clear();
    
    // We need to be careful with OMP and `stored_paths` resizing.
    // Pre-allocate `stored_paths` is done.
    
    #pragma omp parallel
    {
        std::vector<int> local_batch_indices;
        std::vector<float> local_tensor;
        
        #pragma omp for
        for (int k = 0; k < (int)active_games.size(); ++k) {
            int idx = active_games[k];
            Game* g = games[idx];
            Node* node = g->root;
            HexBoard sim_board;
            sim_board.copy_from(g->board);
            
            std::vector<Node*> path;
            path.push_back(node);
            
            while (!node->children.empty() && sim_board.check_winner() == 0) {
                float best_score = -1e9;
                Node* best_child = nullptr;
                int best_action = -1;
                float sqrt_visit = std::sqrt((float)node->visit_count);
                
                for (Node* child : node->children) {
                    float q = 0;
                    if (child->visit_count > 0) q = -child->value_sum / child->visit_count;
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
                } else {
                    break;
                }
            }
            
            // Store path
            stored_paths[idx] = path;
            
            int winner = sim_board.check_winner();
            if (winner != 0) {
                // Terminal. Backpropagate immediately.
                // We need to track the player at each node to assign correct value.
                int p = g->board.current_player;
                
                for (size_t k = 0; k < path.size(); ++k) {
                    Node* n = path[k];
                    n->visit_count++;
                    
                    // Value is +1 if the player at this node WON, -1 if LOST.
                    float v = (winner == p) ? 1.0f : -1.0f;
                    n->value_sum += v;
                    
                    // Determine next player
                    if (k + 1 < path.size()) {
                        // Standard alternation.
                        // Even if Swap was played, our apply_move logic now handles the board state
                        // such that the "next" player is indeed the opponent.
                        // (Swap effectively makes P2 play a P2 stone at a new location).
                        p = (p == RED) ? BLUE : RED;
                    }
                }
            } else {
                // Not terminal, add to batch
                local_batch_indices.push_back(idx);
                
                int my = sim_board.current_player;
                int opp = (my == RED) ? BLUE : RED;
                
                for (int r=0; r<BOARD_SIZE; ++r) {
                    for (int c=0; c<BOARD_SIZE; ++c) {
                        int t = r*BOARD_SIZE + c;
                        int val = sim_board.grid[t];
                        local_tensor.push_back(val == my ? 1.0f : 0.0f);
                    }
                }
                for (int r=0; r<BOARD_SIZE; ++r) {
                    for (int c=0; c<BOARD_SIZE; ++c) {
                        int t = r*BOARD_SIZE + c;
                        int val = sim_board.grid[t];
                        local_tensor.push_back(val == opp ? 1.0f : 0.0f);
                    }
                }
                // Plane 2: Player Color (1.0 for Red/Vertical, 0.0 for Blue/Horizontal)
                // This tells the network which direction to connect.
                float color_val = (my == RED) ? 1.0f : 0.0f;
                for (int i=0; i<NUM_TILES; ++i) local_tensor.push_back(color_val);
            }
        }
        
        #pragma omp critical
        {
            int offset = batch_indices.size();
            batch_indices.insert(batch_indices.end(), local_batch_indices.begin(), local_batch_indices.end());
            
            for (int i=0; i<(int)local_batch_indices.size(); ++i) {
                indices[offset + i] = local_batch_indices[i];
                int base = (offset + i) * 3 * NUM_TILES;
                int local_base = i * 3 * NUM_TILES;
                for (int j=0; j<3*NUM_TILES; ++j) {
                    board_tensor[base + j] = local_tensor[local_base + j];
                }
            }
        }
    }
    
    return batch_indices.size();
}

// Updates the leaf nodes with policy and value, expands them, and backpropagates
void mcts_update(int count, int* indices, float* policies, float* values) {
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        int idx = indices[i];
        float val = values[i]; // Value for the player at the leaf
        
        std::vector<Node*>& path = stored_paths[idx];
        Node* leaf = path.back();
        Game* g = games[idx];
        
        // 1. Re-simulate to get valid moves at the leaf state
        // We do this to avoid storing full board states in every node (memory optimization)
        HexBoard sim_board;
        sim_board.copy_from(g->board);
        for (size_t k=1; k<path.size(); ++k) {
            sim_board.apply_move(path[k]->action);
        }
        
        std::vector<int> moves;
        sim_board.get_valid_moves(moves);
        
        // 2. Expand the leaf node
        float policy_sum = 0.0f;
        for (int move : moves) {
            policy_sum += policies[i * NUM_ACTIONS + move];
        }
        
        if (policy_sum > 1e-6) {
            for (int move : moves) {
                float p = policies[i * NUM_ACTIONS + move] / policy_sum;
                leaf->children.push_back(g->pool.alloc(p, move));
            }
        } else {
            // Fallback to uniform policy if NN output is invalid
            float p = 1.0f / moves.size();
            for (int move : moves) {
                leaf->children.push_back(g->pool.alloc(p, move));
            }
        }
        
        // 3. Backpropagate (Negamax)
        // val is the value for the player at the leaf node.
        // The parent node (who moved to get here) sees this as -val.
        float v = val;
        for (int k = path.size() - 1; k >= 0; --k) {
            path[k]->visit_count++;
            path[k]->value_sum += v;
            v = -v; // Flip perspective for the parent
        }
    }
}

// Play moves for all games based on MCTS counts
// Returns number of finished games
// Writes history to buffers
int play_moves(float* history_boards, float* history_probs, float* history_values, int* finished_count) {
    int finished = 0;
    int num_games = games.size();
    int example_count = 0;
    
    static std::mt19937 serial_rng(global_seed);
    
    // We iterate serially here as we might modify the global games vector or RNG
    // But since we just read/write to specific game instances, we could parallelize if needed.
    // For now, serial is fast enough compared to MCTS/NN.
    
    for (int i = 0; i < num_games; ++i) {
        Game* g = games[i];
        if (g->board.check_winner() != 0) continue; 
        
        Node* root = g->root;
        
        // 1. Calculate Policy from Visit Counts
        std::vector<float> probs(NUM_ACTIONS, 0.0f);
        float sum_visits = 0;
        for (Node* child : root->children) {
            probs[child->action] = child->visit_count;
            sum_visits += child->visit_count;
        }
        
        // Handle case with no visits (should not happen with proper MCTS sims)
        if (sum_visits == 0) {
             std::vector<int> moves;
             g->board.get_valid_moves(moves);
             int action = moves[serial_rng() % moves.size()];
             g->board.apply_move(action);
             g->pool.reset();
             g->root = g->pool.alloc(0, -1);
             continue;
        }
        
        // Normalize
        for (int k=0; k<NUM_ACTIONS; ++k) probs[k] /= sum_visits;
        
        // 2. Select Action (Sampling vs Argmax)
        int chosen_action = -1;
        // Temperature schedule: Explore for first 15 moves, then play strongest
        if (g->board.move_count < 15) {
            float r = std::uniform_real_distribution<float>(0, 1)(serial_rng);
            float cum = 0;
            for (int k=0; k<NUM_ACTIONS; ++k) {
                cum += probs[k];
                if (r <= cum) {
                    chosen_action = k;
                    break;
                }
            }
            if (chosen_action == -1) chosen_action = NUM_ACTIONS - 1;
        } else {
            float max_p = -1;
            for (int k=0; k<NUM_ACTIONS; ++k) {
                if (probs[k] > max_p) {
                    max_p = probs[k];
                    chosen_action = k;
                }
            }
        }
        
        // 3. Store Training Example
        // Board state (from perspective of current player)
        int my = g->board.current_player;
        int opp = (my == RED) ? BLUE : RED;
        
        // We flatten the 3x11x11 planes directly into the history buffer
        for (int r=0; r<BOARD_SIZE; ++r) {
            for (int c=0; c<BOARD_SIZE; ++c) {
                int t = r*BOARD_SIZE + c;
                int val = g->board.grid[t];
                g->history_planes_flat.push_back(val == my ? 1.0f : 0.0f);
            }
        }
        for (int r=0; r<BOARD_SIZE; ++r) {
            for (int c=0; c<BOARD_SIZE; ++c) {
                int t = r*BOARD_SIZE + c;
                int val = g->board.grid[t];
                g->history_planes_flat.push_back(val == opp ? 1.0f : 0.0f);
            }
        }
        // Plane 2: Player Color
        float color_val = (my == RED) ? 1.0f : 0.0f;
        for (int k=0; k<NUM_TILES; ++k) g->history_planes_flat.push_back(color_val);
        
        g->history_probs_flat.insert(g->history_probs_flat.end(), probs.begin(), probs.end());
        g->history_players.push_back(my);
        
        // 4. Apply Move & Reset Tree
        g->board.apply_move(chosen_action);
        
        // For maximum throughput and memory safety, we reset the tree after every move.
        // Preserving the subtree is possible but complex with the NodePool and high parallelism.
        g->pool.reset();
        g->root = g->pool.alloc(0, -1);
        
        // 5. Check Winner & Harvest Data
        int winner = g->board.check_winner();
        if (winner != 0) {
            finished++;
            
            // Backfill the value for all examples in this game
            for (size_t k=0; k<g->history_players.size(); ++k) {
                int p = g->history_players[k];
                // Value is 1.0 if this player won, -1.0 otherwise
                float v = (p == winner) ? 1.0f : -1.0f;
                
                // Copy to output buffers
                int base_board = example_count * 3 * NUM_TILES;
                int base_probs = example_count * NUM_ACTIONS;
                int base_value = example_count;
                
                int src_board = k * 3 * NUM_TILES;
                int src_probs = k * NUM_ACTIONS;
                
                std::memcpy(history_boards + base_board, &g->history_planes_flat[src_board], 3 * NUM_TILES * sizeof(float));
                std::memcpy(history_probs + base_probs, &g->history_probs_flat[src_probs], NUM_ACTIONS * sizeof(float));
                history_values[base_value] = v;
                
                example_count++;
            }
            
            // Immediately reset the game to keep the batch full
            g->reset();
        }
    }
    
    *finished_count = example_count;
    return 0;
}

}
