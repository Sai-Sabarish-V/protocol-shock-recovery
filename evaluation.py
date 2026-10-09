
import random
import torch
import torch.nn.functional as F

from data.dataset import ObjectDataset
from data.referential_game import ReferentialGame
from models.speaker import build_sender
from models.listener import build_listener


# ==========================================
# 1. Configuration
# ==========================================
SEED = 123
NUM_GAMES = 1000

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# 2. Load saved checkpoint
# ==========================================
checkpoint = torch.load(
    "referential_game.pt",
    map_location=device,
    weights_only=False,
)

vocab_size = checkpoint["vocab_size"]
num_candidates = checkpoint["num_candidates"]

print("Vocabulary size:", vocab_size)
print("Number of candidates:", num_candidates)


# ==========================================
# 3. Rebuild speaker and listener
# ==========================================
sender = build_sender(
    vocab_size=vocab_size,
    embed_dim=32,
    hidden_size=128,
    object_embedding_dim=16,
)

receiver = build_listener(
    vocab_size=vocab_size,
    receiver_embed_dim=32,
    receiver_hidden_size=128,
    candidate_embedding_dim=16,
)


# ==========================================
# 4. Load trained weights
# ==========================================
state = checkpoint["model_state_dict"]

sender_weights = {
    key.removeprefix("sender."): value
    for key, value in state.items()
    if key.startswith("sender.")
}

receiver_weights = {
    key.removeprefix("receiver."): value
    for key, value in state.items()
    if key.startswith("receiver.")
}

sender.load_state_dict(sender_weights)
receiver.load_state_dict(receiver_weights)

sender.to(device)
receiver.to(device)

sender.eval()
receiver.eval()


# ==========================================
# 5. Create evaluation game
# ==========================================
dataset = ObjectDataset()

game = ReferentialGame(
    dataset=dataset,
    num_candidates=num_candidates,
)


# ==========================================
# 6. Evaluate the combined system
# ==========================================
correct = 0
total_loss = 0.0
total = 0
printed_shapes = False

with torch.no_grad():

    for _ in range(NUM_GAMES):

        sample = game.sample_game()

        target = (
            sample["target"]
            .unsqueeze(0)
            .to(device)
        )

        candidates = (
            sample["candidates"]
            .unsqueeze(0)
            .to(device)
        )

        label = torch.tensor(
            [sample["target_index"]],
            dtype=torch.long,
            device=device,
        )

        # Generate message from the target object.
        message = sender(target)

        # Receiver outputs scores at every message timestep.
        all_scores = receiver(message, candidates)

        # Select the final timestep, after processing the
        # complete message, including the appended EOS token.
        scores = all_scores[:, -1, :]

        if not printed_shapes:
            print("\n--- Tensor shapes ---")
            print("Target:", target.shape)
            print("Message:", message.shape)
            print("Candidates:", candidates.shape)
            print("All receiver scores:", all_scores.shape)
            print("Final scores:", scores.shape)
            print("Label:", label.shape)
            printed_shapes = True

        # Validate dimensions before calculating loss.
        if scores.shape != (1, num_candidates):
            raise ValueError(
                f"Expected scores shape {(1, num_candidates)}, "
                f"but received {tuple(scores.shape)}"
            )

        # Calculate classification loss.
        loss = F.cross_entropy(scores, label)

        # Select the candidate with the highest score.
        prediction = scores.argmax(dim=1)

        correct += (prediction == label).sum().item()
        total_loss += loss.item()
        total += 1


# ==========================================
# 7. Display results
# ==========================================
print("\n--- Evaluation Results ---")
print("Games evaluated:", total)

if total > 0:
    accuracy = 100.0 * correct / total
    average_loss = total_loss / total
    random_accuracy = 100.0 / num_candidates

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average loss: {average_loss:.4f}")
    print(f"Random-guess accuracy: {random_accuracy:.2f}%")

print("\nEvaluation complete.")
