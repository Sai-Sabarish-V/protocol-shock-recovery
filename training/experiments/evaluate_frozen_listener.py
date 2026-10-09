
import random
import torch
import torch.nn.functional as F

from data.dataset import ObjectDataset
from data.referential_game import ReferentialGame
from models.speaker import build_sender
from models.listener import build_listener

SEED = 456
NUM_GAMES = 1000

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the original speaker checkpoint.
checkpoint = torch.load(
    "referential_game.pt",
    map_location=device,
    weights_only=False,
)

# Load the best fresh listener.
listener_checkpoint = torch.load(
    "frozen_listener.pt",
    map_location=device,
    weights_only=False,
)

sender = build_sender(
    vocab_size=checkpoint["vocab_size"],
    embed_dim=32,
    hidden_size=128,
    object_embedding_dim=16,
)

receiver = build_listener(
    vocab_size=listener_checkpoint["vocab_size"],
    receiver_embed_dim=32,
    receiver_hidden_size=128,
    candidate_embedding_dim=16,
)

sender_state = {
    k.removeprefix("sender."): v
    for k, v in checkpoint["model_state_dict"].items()
    if k.startswith("sender.")
}

sender.load_state_dict(sender_state)
receiver.load_state_dict(
    listener_checkpoint["receiver_state_dict"]
)

sender.to(device).eval()
receiver.to(device).eval()

for parameter in sender.parameters():
    parameter.requires_grad = False

dataset = ObjectDataset()
num_candidates = listener_checkpoint["num_candidates"]

game = ReferentialGame(
    dataset=dataset,
    num_candidates=num_candidates,
)

correct = 0
loss_total = 0.0

with torch.no_grad():
    for _ in range(NUM_GAMES):
        sample = game.sample_game()

        target = sample["target"].unsqueeze(0).to(device)
        candidates = sample["candidates"].unsqueeze(0).to(device)
        labels = torch.tensor(
            [sample["target_index"]],
            dtype=torch.long,
            device=device,
        )

        message = sender(target)
        all_scores = receiver(message, candidates)
        scores = all_scores[:, -1, :]

        loss_total += F.cross_entropy(scores, labels).item()
        correct += (
            scores.argmax(dim=1) == labels
        ).sum().item()

print("Games:", NUM_GAMES)
print(f"Accuracy: {100 * correct / NUM_GAMES:.2f}%")
print(f"Average loss: {loss_total / NUM_GAMES:.4f}")
print(f"Random baseline: {100 / num_candidates:.2f}%")
