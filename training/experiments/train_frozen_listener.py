
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.dataset import ObjectDataset
from data.referential_game import ReferentialGame
from models.speaker import build_sender
from models.listener import build_listener


# ==========================================
# 1. Configuration
# ==========================================
SEED = 123
BATCH_SIZE = 64
EPOCHS = 30
GAMES_PER_EPOCH = 5000
VALIDATION_GAMES = 1000
LEARNING_RATE = 1e-3

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Dataset of freshly sampled games
# ==========================================
class GameDataset(Dataset):
    def __init__(self, dataset, num_candidates, num_games):
        self.dataset = dataset
        self.game = ReferentialGame(
            dataset=dataset,
            num_candidates=num_candidates,
        )
        self.num_games = num_games

    def __len__(self):
        return self.num_games

    def __getitem__(self, index):
        sample = self.game.sample_game()

        return (
            sample["target"],
            sample["candidates"],
            torch.tensor(
                sample["target_index"],
                dtype=torch.long,
            ),
        )


# ==========================================
# 3. Load the existing speaker checkpoint
# ==========================================
checkpoint = torch.load(
    "referential_game.pt",
    map_location=device,
    weights_only=False,
)

vocab_size = checkpoint["vocab_size"]
num_candidates = checkpoint["num_candidates"]

objects = ObjectDataset()

print("Objects:", len(objects))
print("Vocabulary size:", vocab_size)
print("Candidates:", num_candidates)


# ==========================================
# 4. Build and freeze the speaker
# ==========================================
sender = build_sender(
    vocab_size=vocab_size,
    embed_dim=32,
    hidden_size=128,
    object_embedding_dim=16,
)

state = checkpoint["model_state_dict"]

sender_weights = {
    key.removeprefix("sender."): value
    for key, value in state.items()
    if key.startswith("sender.")
}

sender.load_state_dict(sender_weights)
sender.to(device)

# Freeze every speaker parameter.
for parameter in sender.parameters():
    parameter.requires_grad = False

# Keep messages deterministic during listener training.
sender.eval()

print("\nSpeaker loaded and frozen.")
print(
    "Trainable speaker parameters:",
    sum(p.numel() for p in sender.parameters() if p.requires_grad),
)


# ==========================================
# 5. Create a completely fresh listener
# ==========================================
receiver = build_listener(
    vocab_size=vocab_size,
    receiver_embed_dim=32,
    receiver_hidden_size=128,
    candidate_embedding_dim=16,
)

receiver.to(device)

print(
    "Trainable listener parameters:",
    sum(p.numel() for p in receiver.parameters() if p.requires_grad),
)


# ==========================================
# 6. Prepare training and validation data
# ==========================================
train_dataset = GameDataset(
    objects,
    num_candidates,
    GAMES_PER_EPOCH,
)

validation_dataset = GameDataset(
    objects,
    num_candidates,
    VALIDATION_GAMES,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================
# 7. Optimize ONLY the listener
# ==========================================
optimizer = torch.optim.Adam(
    receiver.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss()

best_accuracy = -1.0

print("\nStarting frozen-speaker listener training...\n")


# ==========================================
# 8. Training loop
# ==========================================
for epoch in range(EPOCHS):

    receiver.train()
    sender.eval()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for targets, candidates, labels in train_loader:

        targets = targets.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        # Generate messages without updating the speaker.
        with torch.no_grad():
            messages = sender(targets)

        # Receiver produces scores at every timestep.
        all_scores = receiver(messages, candidates)

        # Classify after processing the complete sequence.
        scores = all_scores[:, -1, :]

        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = scores.argmax(dim=1)

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            predictions == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = 100.0 * train_correct / train_total


    # ======================================
    # 9. Validation
    # ======================================
    receiver.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for targets, candidates, labels in validation_loader:

            targets = targets.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            # Same frozen, deterministic speaker.
            messages = sender(targets)

            all_scores = receiver(messages, candidates)
            scores = all_scores[:, -1, :]

            loss = criterion(scores, labels)
            predictions = scores.argmax(dim=1)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (
                predictions == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = 100.0 * val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2f}%"
    )

    # Save the best listener, not the speaker.
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            {
                "receiver_state_dict": receiver.state_dict(),
                "vocab_size": vocab_size,
                "num_candidates": num_candidates,
                "epoch": epoch + 1,
                "validation_accuracy": val_accuracy,
                "speaker_checkpoint": "referential_game.pt",
            },
            "frozen_listener.pt",
        )


# ==========================================
# 10. Final summary
# ==========================================
print("\nTraining complete.")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print("Best listener saved to frozen_listener.pt")
print("Original speaker checkpoint was not modified.")
