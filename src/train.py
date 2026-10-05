import torch
from torch import nn
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]/"Data"

def load_sentences():
    sentences = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        file_count = 0

        with file_path.open(encoding="utf-8") as file:
            for line in file:
                words = re.findall(r"[a-z]+", line.lower())
                if words:
                    sentences.append(words)
                    file_count += 1

        print(f"{file_path.name}: {file_count} sentences")

    return sentences

def Build_vocab(Sentences):
    unique_words = set()

    for sentence in Sentences:
        for word in sentence:
            unique_words.add(word)
	
    Words = sorted(unique_words)

    Word_to_Id = {}
    Id_to_Word = {}
    Word_id=0

    for word in Words:
        Word_to_Id[word] = Word_id
        Word_id += 1

    for word in Word_to_Id:
        Word_id = Word_to_Id[word]
        Id_to_Word[Word_id] = word
	
    return Word_to_Id,Id_to_Word

def generate_Training_pairs(Sentences, Word_to_Id, Window_size=2):
    Center_Ids = []
    Context_Ids = []

    for sentence in Sentences:
        Sentence_Ids = []

        for word in sentence:
            Word_Id = Word_to_Id[word]
            Sentence_Ids.append(Word_Id)

        for center_position in range(len(Sentence_Ids)):
            center_word_id = Sentence_Ids[center_position]

            first_context_position = center_position - Window_size
            if first_context_position < 0:
                first_context_position = 0

            last_context_position = center_position + Window_size + 1
            if last_context_position > len(Sentence_Ids):
                last_context_position = len(Sentence_Ids)

            for context_position in range(
                first_context_position,
                last_context_position,
            ):
                if context_position != center_position:
                    context_word_id = Sentence_Ids[context_position]
                    Center_Ids.append(center_word_id)
                    Context_Ids.append(context_word_id)

    if len(Center_Ids) == 0:
        raise ValueError("No training pairs were created.")

    Center_tensor = torch.tensor(Center_Ids, dtype=torch.long)
    Context_tensor = torch.tensor(Context_Ids, dtype=torch.long)

    return Center_tensor, Context_tensor


class SkipGram(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings= vocab_size,
            embedding_dim= embedding_size,
        )
        self.context_predictor = nn.Linear(
            in_features=embedding_size,
            out_features=vocab_size,
        )
    def forward(self,center_word_Ids):
        word_vectors = self.embeddings(center_word_Ids)
        context_scores = self.context_predictor(word_vectors)
        return context_scores

def train_model(config, center_ids, context_ids, vocabulary_size):
    embedding_size = config.get("embedding_size", 32)
    number_of_epochs = config.get("number_of_epochs", 50)
    batch_size = config.get("batch_size", 128)
    learning_rate = config.get("learning_rate", 0.01)

    model = SkipGram(
        vocab_size=vocabulary_size,
        embedding_size=embedding_size,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()
    number_of_training_pairs = len(center_ids)

    for epoch_number in range(number_of_epochs):
        shuffled_indexes = torch.randperm(number_of_training_pairs)
        total_loss = 0.0
        number_of_batches = 0

        for batch_start in range(
            0,
            number_of_training_pairs,
            batch_size,
        ):
            batch_end = batch_start + batch_size
            batch_indexes = shuffled_indexes[batch_start:batch_end]

            batch_center_ids = center_ids[batch_indexes]
            batch_context_ids = context_ids[batch_indexes]

            predicted_context_scores = model(batch_center_ids)
            loss = loss_function(predicted_context_scores, batch_context_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            number_of_batches = number_of_batches + 1

        if epoch_number == 0 or (epoch_number + 1) % 10 == 0:
            average_loss = total_loss / number_of_batches
            print(
                f"Epoch {epoch_number + 1}: "
                f"loss = {average_loss:.5f}"
            )

    return model


def Train(config):
    sentences = load_sentences()
    word_to_id, id_to_word = Build_vocab(sentences)

    print(f"Loaded {len(sentences)} sentences.")
    print(f"Vocabulary contains {len(word_to_id)} unique words.")

    center_ids, context_ids = generate_Training_pairs(
        Sentences=sentences,
        Word_to_Id=word_to_id,
        Window_size=2,
    )
    print(f"Created {len(center_ids)} training pairs.")

    model = train_model(
        config=config,
        center_ids=center_ids,
        context_ids=context_ids,
        vocabulary_size=len(word_to_id),
    )

    models_folder = Path(__file__).resolve().parents[1] / "Models"
    models_folder.mkdir(parents=True, exist_ok=True)
    save_path = models_folder / "word_embeddings.pt"

    saved_data = {
        "model_state_dict": model.state_dict(),
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }

    torch.save(saved_data, save_path)
    print(f"Saved the trained model to: {save_path}")
