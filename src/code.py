import ast
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from train import SkipGram


MODEL_PATH = Path(__file__).resolve().parents[1] / "Models" / "word_embeddings.pt"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first; the saved model file is missing.")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    word_to_id = checkpoint["word_to_id"]
    id_to_word = checkpoint["id_to_word"]

    saved_weights = checkpoint["model_state_dict"]
    embedding_size = saved_weights["embeddings.weight"].shape[1]

    model = SkipGram(
        vocab_size=len(word_to_id),
        embedding_size=embedding_size,
    )
    model.load_state_dict(saved_weights)
    model.eval()

    return model, word_to_id, id_to_word

def Encode(Data:str):
    model,word_to_id, id_to_word = load_model()
    
    words = re.findall(r"[a-z]+", Data.lower())
    word_ids = []

    for word in words:
        if word in word_to_id:
            word_id = word_to_id[word]
            word_ids.append(word_id)

    if len(word_ids) == 0:
        raise ValueError("No words from the input were found in the model vocabulary.")
    input_ids = torch.tensor(word_ids,dtype=torch.long)

    with torch.no_grad():
        word_vectors = model.embeddings(input_ids)
        text_vector = word_vectors.mean(dim=0)
    return text_vector

def Decode(Data:str):
    model, word_to_id, id_to_word = load_model()

    opening_bracket = Data.find("[")
    closing_bracket = Data.rfind("]")

    if opening_bracket == -1 or closing_bracket == -1:
        raise ValueError("Enter a vector such as [0.1, 0.2, 0.3].")

    vector_text = Data[opening_bracket : closing_bracket + 1]

    try:
        vector_values = ast.literal_eval(vector_text)
    except (SyntaxError, ValueError) as error:
        raise ValueError("The vector must contain valid numbers in a list.") from error

    if not isinstance(vector_values, list):
        raise ValueError("The vector must be entered as a list of numbers.")

    query_vector = torch.tensor(vector_values, dtype=torch.float32)
    expected_size = model.embeddings.embedding_dim

    if query_vector.ndim != 1 or len(query_vector) != expected_size:
        raise ValueError(
            f"Expected a vector with {expected_size} numbers, "
            f"but received {len(query_vector)}."
        )

    all_word_vectors = model.embeddings.weight.detach()
    normalized_word_vectors = F.normalize(all_word_vectors, p=2, dim=1)
    normalized_query_vector = F.normalize(query_vector, p=2, dim=0)
    similarity_scores = torch.matmul(
        normalized_word_vectors,
        normalized_query_vector,
    )

    number_of_results = min(5, len(word_to_id))
    best_scores, best_word_ids = torch.topk(
        similarity_scores,
        k=number_of_results,
    )

    decoded_words = []

    for result_position in range(number_of_results):
        word_id = best_word_ids[result_position].item()
        word = id_to_word[word_id]
        score = best_scores[result_position].item()
        decoded_words.append(f"{word} ({score:.3f})")

    return "Nearest words: " + ", ".join(decoded_words)