# embedding_api.py

import time
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import List, Dict, Optional
from fastapi.responses import JSONResponse
from urllib.parse import unquote_plus

app = FastAPI()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Predefined instructions with examples
INSTRUCTIONS = {
    "play_previous": {
        "name": "播放上一句",
        "examples": [
            "上一句",
            "播放上一句",
            "播放刚才那句",
            "回退一句",
            "返回上一句",
            "重播前一句"
        ]
    },
    "play_next": {
        "name": "播放下一句",
        "examples": [
            "下一句",
            "播放下一句",
            "播放后一句",
            "前进一句",
            "继续播放"
        ]
    },
    "pause": {
        "name": "暂停播放",
        "examples": [
            "暂停",
            "停止播放",
            "暂停播放",
            "先停一下"
        ]
    },
    "speed_075": {
        "name": "设置0.75倍速",
        "examples": [
            "0.75倍速",
            "放慢速度",
            "速度调为0.75倍",
            "调慢一点"
        ]
    },
    "speed_normal": {
        "name": "恢复正常速度",
        "examples": [
            "正常速度",
            "恢复原速度",
            "恢复正常速度",
            "原速播放"
        ]
    }
}

# Cache for instruction embeddings
instruction_embeddings = {}
instruction_texts = {}

# Function to get the last token's embedding
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Response format for embedding
class EmbeddingResponse(BaseModel):
    elapsed_time: float
    embedding: List[float]

# Response format for query
class QueryResponse(BaseModel):
    elapsed_time: float
    matched_instruction: Optional[str] = None
    similarity_score: Optional[float] = None

# Initialize instruction embeddings
def initialize_instruction_embeddings():
    all_examples = []
    example_to_instruction = {}
    
    # Collect all examples
    for instruction_id, instruction_data in INSTRUCTIONS.items():
        for example in instruction_data["examples"]:
            all_examples.append(example)
            example_to_instruction[example] = instruction_id
    
    # Compute embeddings in batch
    batch = tokenizer(all_examples, return_tensors="pt", padding=True, truncation=True, max_length=8192)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Organize embeddings by instruction
    for example, embedding in zip(all_examples, embeddings):
        instruction_id = example_to_instruction[example]
        if instruction_id not in instruction_embeddings:
            instruction_embeddings[instruction_id] = []
        instruction_embeddings[instruction_id].append(embedding)
        
        if instruction_id not in instruction_texts:
            instruction_texts[instruction_id] = []
        instruction_texts[instruction_id].append(example)

# Initialize embeddings on startup
initialize_instruction_embeddings()

@app.get("/embed/{text}")
async def get_embedding(text: str):
    # URL decode the text
    decoded_text = unquote_plus(text)
    start_time = time.time()

    # Tokenize
    batch = tokenizer([decoded_text], return_tensors="pt", padding=True, truncation=True, max_length=8192)
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    elapsed_time = time.time() - start_time
    return EmbeddingResponse(
        elapsed_time=elapsed_time,
        embedding=embeddings[0].cpu().tolist(),
    )

@app.get("/query/{text}")
async def query_instruction(text: str):
    # URL decode the text
    decoded_text = unquote_plus(text)
    start_time = time.time()

    # Get embedding for query text
    batch = tokenizer([decoded_text], return_tensors="pt", padding=True, truncation=True, max_length=8192)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        query_embedding = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # Find best matching instruction
    best_score = -1
    best_instruction = None
    
    for instruction_id, embeddings_list in instruction_embeddings.items():
        # Calculate similarity with all examples of this instruction
        similarities = torch.matmul(query_embedding, torch.stack(embeddings_list).T)
        max_similarity = similarities.max().item()
        
        if max_similarity > best_score:
            best_score = max_similarity
            best_instruction = instruction_id

    elapsed_time = time.time() - start_time
    
    # Use a threshold to determine if the match is good enough
    SIMILARITY_THRESHOLD = 0.75
    if best_score < SIMILARITY_THRESHOLD:
        return QueryResponse(
            elapsed_time=elapsed_time,
            matched_instruction=None,
            similarity_score=best_score
        )
    
    return QueryResponse(
        elapsed_time=elapsed_time,
        matched_instruction=INSTRUCTIONS[best_instruction]["name"],
        similarity_score=best_score
    )
