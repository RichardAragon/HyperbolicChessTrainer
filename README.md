# HyperbolicChessTrainer
Auto trains LLM models to play chess by having them simulate the game board in hyperbolic space
# Hyperbolic Chess Trainer

## Overview

**Hyperbolic Chess Trainer** is an experimental AI-driven chess playing and training tool. It utilizes hyperbolic geometry-inspired embeddings and neural networks to evaluate chessboard states, make intelligent moves, and store past games for improving decision-making. The project is released under the MIT license, promoting open collaboration and usage.

## Features

- **Hyperbolic Embeddings**: Encodes chessboard states using hyperbolic transformations for efficient and insightful move evaluations.
- **Memory Module**: A custom memory system retrieves historically relevant board states based on hyperbolic similarity metrics.
- **Language Model Integration**: Leverages a transformer-based language model to generate moves and interpret game contexts.
- **Chess Gameplay**: Plays full chess games with up to 100 moves, adhering to legal game rules and maintaining a rich memory of past moves.
- **Customizability**: Adjustable memory size, embedding dimensions, and model checkpoints to suit different computational needs.

## Prerequisites

- Python 3.8+
- PyTorch 1.9+ with CUDA (optional for GPU acceleration)
- Required Python libraries:
  - `torch`
  - `numpy`
  - `transformers`
  - `python-chess`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RichardAragon/HyperbolicChessTrainer.git
   cd HyperbolicChessTrainer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure a GPU environment:
   Ensure PyTorch with CUDA is installed for faster computations:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   ```

## Usage

Run the `main.py` script to initiate a chess game:
```bash
python main.py
```

The program initializes a chess agent using hyperbolic embeddings and a pre-trained language model. The agent will play a complete game of chess, logging moves and game progress.

### Configuration

- **Model Checkpoint**: Adjust the pre-trained language model by modifying the `model_checkpoint` parameter in `main.py`.
- **Device**: The script automatically detects and uses a GPU if available. To force a specific device, set the `device` parameter (`cpu` or `cuda`).
- **Memory Size**: Customize the memory buffer size in the `HyperbolicChessMemory` class (`memory_size` argument).
- **Embedding Dimensions**: Modify the `embedding_dim` argument in the `HyperbolicChessEmbedding` class to adjust embedding vector size.

## Project Structure

- **`main.py`**: Entry point for running the chess agent.
- **`HyperbolicChessAgent`**: Core class managing chess gameplay and AI decision-making.
- **`HyperbolicChessMemory`**: Memory module for storing and retrieving hyperbolic embeddings of past board states.
- **`HyperbolicChessEmbedding`**: Neural network for generating hyperbolic embeddings from board states.

## Known Limitations

- **Resource Intensive**: Large model checkpoints may require significant computational resources.
- **Experimental Design**: The current implementation is a prototype; accuracy and performance can vary.

## License

Hyperbolic Chess Trainer is licensed under the [MIT License](LICENSE).

---

## Contribution

Contributions are welcome! If you'd like to report a bug, suggest a feature, or submit a pull request, please visit the [GitHub repository](https://github.com/yourusername/HyperbolicChessTrainer).

## Acknowledgments

- **PyTorch** for the neural network framework.
- **Transformers by Hugging Face** for the language model.
- **python-chess** for chessboard representation and move validation.

---

**Happy Chess Training!** 🎮♟️
