from pathlib import Path
import torch
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

def Decode(Data:str):
    pass