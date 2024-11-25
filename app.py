import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperbolicChessMemory:
    def __init__(self, memory_size=100, embedding_dim=256, device="cpu"):
        self.device = device
        self.memory_tensor = torch.zeros((memory_size, embedding_dim), device=self.device)
        self.memory_mask = torch.zeros(memory_size, dtype=torch.bool, device=self.device)
        self.current_index = 0
        logger.info(f"[Memory] Initialized with size {memory_size} and embedding dim {embedding_dim} on {device}")

    def store_move(self, hyperbolic_move_embedding: torch.Tensor):
        self.memory_tensor[self.current_index] = hyperbolic_move_embedding
        self.memory_mask[self.current_index] = True
        logger.info(f"[Memory] Stored move at index {self.current_index}")
        self.current_index = (self.current_index + 1) % self.memory_tensor.shape[0]

    def retrieve_relevant_moves(self, query_embedding: torch.Tensor, top_k=5):
        # Find valid memories
        valid_memories = self.memory_tensor[self.memory_mask]
        
        # If no valid memories, return empty list
        if len(valid_memories) == 0:
            logger.info("[Memory] No valid memories to retrieve.")
            return []
        
        # Ensure query_embedding is a 1D tensor
        query_embedding = query_embedding.squeeze()
        
        # Debug: Print shapes and types
        logger.debug(f"[Memory] Query embedding shape: {query_embedding.shape}")
        logger.debug(f"[Memory] Valid memories shape: {valid_memories.shape}")
        
        # Compute similarities
        try:
            similarities = F.cosine_similarity(query_embedding.unsqueeze(0), valid_memories)
        except Exception as e:
            logger.error(f"[Memory] Similarity computation failed: {e}")
            return []
        
        # Debug: Print similarities
        logger.debug(f"[Memory] Similarities: {similarities}")
        
        # If only one memory, return it
        if len(similarities) == 1:
            logger.info("[Memory] Only one memory available.")
            return valid_memories
        
        # Sort similarities and get indices
        sorted_indices = torch.argsort(similarities, descending=True)
        
        # Safely select top-k indices
        top_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        
        logger.info(f"[Memory] Retrieved top {len(top_indices)} relevant moves.")
        
        return valid_memories[top_indices]

class HyperbolicChessEmbedding(nn.Module):
    def __init__(self, board_size=8, embedding_dim=256):
        super().__init__()
        self.piece_embeddings = nn.Embedding(13, embedding_dim)
        self.position_embeddings = nn.Embedding(64, embedding_dim)
        self.hyperbolic_transform = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        logger.info("[Embedding] HyperbolicChessEmbedding initialized.")

    def poincare_embedding(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-5)
        return x / norm * torch.tanh(norm) / norm

    def forward(self, board: chess.Board) -> torch.Tensor:
        device = next(self.parameters()).device
        board_tensor = torch.zeros(64, dtype=torch.long, device=device)
        piece_mapping = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece_mapping.get(piece.piece_type, 0)
                color_offset = 6 if piece.color == chess.BLACK else 0
                board_tensor[square] = piece_type + color_offset
        
        piece_embeds = self.piece_embeddings(board_tensor)
        position_embeds = self.position_embeddings(torch.arange(64, device=device))
        combined_embeds = torch.cat([piece_embeds, position_embeds], dim=-1)
        hyperbolic_embed = self.hyperbolic_transform(combined_embeds)
        
        # Mean pooling to create a single board embedding
        pooled_embed = hyperbolic_embed.mean(dim=0)
        
        return self.poincare_embedding(pooled_embed)

class HyperbolicChessAgent:
    def __init__(self, model_checkpoint="HuggingFaceTB/SmolLM-135M", device="cpu"):
        self.device = device
        logger.info(f"[Agent] Initializing with model checkpoint '{model_checkpoint}' on {self.device}")
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(self.device)
            logger.info("[Agent] Language model loaded successfully.")
        except Exception as e:
            logger.error(f"[Error] Failed to load language model: {e}")
            sys.exit(1)
        
        # Set pad_token if not set
        if self.tokenizer.pad_token is None:
            logger.info("[Agent] Pad token not found. Setting pad_token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.language_model.config.pad_token_id = self.tokenizer.eos_token_id

        # Initialize chess components
        self.hyperbolic_embedding = HyperbolicChessEmbedding().to(self.device)
        self.memory_module = HyperbolicChessMemory(device=self.device)
        self.board = chess.Board()
        logger.info("[Agent] Chess components initialized.")

    def generate_move(self, temperature=0.8, top_k=50) -> str:
        logger.info("[Agent] Generating move...")
        board_embedding = self.hyperbolic_embedding(self.board).to(self.device)
        
        # Retrieve historical moves, but handle all cases
        historical_moves = self.memory_module.retrieve_relevant_moves(board_embedding)
        
        legal_moves = list(self.board.legal_moves)
        move_uci_list = [move.uci() for move in legal_moves]
        
        prompt = f"Chess board state: {self.board.fen()}\n"
        prompt += "Possible moves: " + ", ".join(move_uci_list) + "\n"
        
        # More explicit handling of historical context
        if len(historical_moves) > 0:
            prompt += f"Historical context: {len(historical_moves)} previous moves considered.\n"
        
        prompt += "Choose the best move (in UCI format):"
        logger.debug(f"[Agent] Prompt:\n{prompt}")

        # Tokenize input
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            attention_mask = inputs["attention_mask"]
        except Exception as e:
            logger.error(f"[Error] Tokenization failed: {e}")
            return np.random.choice(move_uci_list)

        # Generate output
        try:
            outputs = self.language_model.generate(
                inputs["input_ids"], 
                attention_mask=attention_mask, 
                max_length=inputs["input_ids"].size(1) + 5,
                num_return_sequences=1,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"[Agent] Generated text: {generated_text}")
        except Exception as e:
            logger.error(f"[Error] Failed to generate move: {e}")
            return np.random.choice(move_uci_list)

        # Extract move from generated text
        for move in move_uci_list:
            if move in generated_text:
                logger.info(f"[Agent] Move selected: {move}")
                return move

        # Fallback: random legal move
        fallback_move = np.random.choice(move_uci_list)
        logger.warning(f"[Agent] Fallback move selected: {fallback_move}")
        return fallback_move

    def play_game(self, max_moves=100):
        logger.info("[Agent] Starting game...")
        move_count = 0
        while not self.board.is_game_over() and move_count < max_moves:
            logger.info(f"\n[Game] Move {move_count + 1}:")
            move = self.generate_move()
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move in self.board.legal_moves:
                    self.board.push(chess_move)
                    logger.info(f"[Game] Move played: {move}")
                else:
                    logger.warning(f"[Warning] Illegal move attempted: {move}. Skipping.")
                    continue
            except ValueError:
                logger.warning(f"[Warning] Invalid UCI move: {move}. Skipping.")
                continue
            
            # Update memory with the new board state
            with torch.no_grad():
                move_embedding = self.hyperbolic_embedding(self.board)
                pooled_embedding = move_embedding.mean(dim=0)  # Mean pooling
                self.memory_module.store_move(pooled_embedding)
            move_count += 1
            logger.info(f"[Game] Current Board:\n{self.board}")

        logger.info("\n[Game] Game over.")
        logger.info(f"Result: {self.board.result()}")
        return self.board.result()

def main():
    logger.info("[Main] Hyperbolic Chess Agent Starting...")
    torch.manual_seed(42)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Main] Using device: {device}")
    
    # Initialize agent with a smaller model for testing
    model_checkpoint = "HuggingFaceTB/SmolLM-360M"  # Change to "HuggingFaceTB/SmolLM2-1.7B" if your system can handle it
    agent = HyperbolicChessAgent(model_checkpoint=model_checkpoint, device=device)
    
    # Play the game
    result = agent.play_game()
    logger.info(f"[Main] Game Result: {result}")

if __name__ == "__main__":
    main()
