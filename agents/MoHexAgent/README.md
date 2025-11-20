# MoHex Agent

This agent interfaces with MoHex, a strong Hex-playing program that uses Monte Carlo Tree Search.

## Installation

### Option 1: Download Pre-built Binary (Easiest)

1. **Download Benzene** (MoHex is part of the Benzene suite):
   - Visit: https://github.com/cgao3/benzene-vanilla-cmake
   - Or the original: https://github.com/cgao3/benzene-vanilla

2. **For Windows:**
   - Download a pre-compiled binary if available
   - Or use WSL (Windows Subsystem for Linux) to compile

3. **For Linux/MacOS:**
   - Download or compile from source (see Option 2)

### Option 2: Compile from Source

#### Prerequisites:
- CMake (>= 3.10)
- C++ compiler (GCC, Clang, or MSVC)
- Boost libraries (>= 1.58)
- Git

#### Linux/WSL:
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install cmake g++ libboost-all-dev git

# Clone Benzene
git clone https://github.com/cgao3/benzene-vanilla-cmake.git
cd benzene-vanilla-cmake

# Build
mkdir build
cd build
cmake ..
make

# The mohex binary will be in build/src/mohex/
```

#### MacOS:
```bash
# Install dependencies via Homebrew
brew install cmake boost git

# Clone and build (same as Linux)
git clone https://github.com/cgao3/benzene-vanilla-cmake.git
cd benzene-vanilla-cmake
mkdir build
cd build
cmake ..
make
```

#### Windows (via Visual Studio):
```powershell
# Install dependencies via vcpkg
vcpkg install boost

# Clone repository
git clone https://github.com/cgao3/benzene-vanilla-cmake.git
cd benzene-vanilla-cmake

# Build with CMake
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Option 3: Use Docker (Alternative)

```bash
# Pull or build a Docker image with Benzene/MoHex
docker pull <benzene-image>
docker run -it <benzene-image> mohex
```

## Setup

After installation, place the `mohex` executable in one of these locations:

1. **In your system PATH** (recommended)
2. **In the MoHexAgent directory**: `agents/MoHexAgent/mohex`
3. **Specify the path** when creating the agent:
   ```python
   agent = MoHexAgent(colour, mohex_path="/path/to/mohex")
   ```

## Usage

### Command Line:
```bash
python Hex.py -p1 "agents.MCTSAgent.MCTSAgent MCTSAgent" -p2 "agents.MoHexAgent.MoHexAgent MoHexAgent"
```

### With GUI:
```bash
python HexGUI.py -p1 "agents.MCTSAgent.MCTSAgent MCTSAgent" -p2 "agents.MoHexAgent.MoHexAgent MoHexAgent"
```

### Custom Time Limit:
The default time limit is 5 seconds per move. To change it, you'll need to modify `MoHexAgent.py`:
```python
agent = MoHexAgent(colour, time_limit=10.0)  # 10 seconds per move
```

## GTP Protocol

MoHex uses the Go Text Protocol (GTP), which is also used by Hex engines. The agent automatically:
- Converts between board coordinates and GTP notation
- Handles the RED/BLUE color mapping to GTP's black/white
- Manages the MoHex process lifecycle

## Troubleshooting

### "MoHex executable not found"
- Ensure MoHex is installed and in your PATH, or specify the path explicitly

### Process hangs or timeouts
- Increase the time limit
- Check that MoHex is responding to GTP commands

### Coordinate conversion errors
- The agent uses standard GTP notation (a-k for columns, 1-11 for rows)
- Ensure your board size matches (default is 11x11)

## References

- Benzene Project: https://github.com/cgao3/benzene-vanilla-cmake
- GTP Specification: https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
- MoHex Paper: Huang, Broderick, et al. "MoHex 2.0: a pattern-based MCTS Hex player."
