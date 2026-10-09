
import random
import torch
import egg.core as core
import argparse
from torch.utils.data import Dataset, DataLoader

from data.dataset import ObjectDataset
from data.referential_game import ReferentialGame
from models.speaker import build_sender
from models.listener import build_listener
from training.loss import referential_loss



BATCH_SIZE = 64
EPOCHS = 50

NUM_CANDIDATES = 8
GAMES_PER_EPOCH = 10_000
VALIDATION_GAMES = 1_000

VOCAB_SIZE = 12
SEED = 42


# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


class ReferentialGameBatchDataset(Dataset):
    """
    Adapts ReferentialGame to PyTorch DataLoader.
    """

    def __init__(self, game, num_games):
        self.game = game
        self.num_games = num_games

    def __len__(self):
        return self.num_games

    def __getitem__(self, index):
        sample = self.game.sample_game()

        target = sample["target"]

        target_index = torch.tensor(
            sample["target_index"],
            dtype=torch.long,
        )

        candidates = sample["candidates"]

        return target, target_index, candidates





def loss_wrapper(
    sender_input,
    message,
    receiver_input,
    receiver_output,
    labels,
    aux_input,
):
    return referential_loss(
        receiver_output,
        labels,
    )




def main():

    # Choose the available device.
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    print(f"Device: {device}")

    # Load objects.
    objects = ObjectDataset()

    print(f"Number of objects: {len(objects)}")
    print(
        "Object feature shape:",
        objects[0]["features"].shape,
    )

    # Create training and validation games.
    train_game = ReferentialGame(
        dataset=objects,
        num_candidates=NUM_CANDIDATES,
    )

    validation_game = ReferentialGame(
        dataset=objects,
        num_candidates=NUM_CANDIDATES,
    )

    train_dataset = ReferentialGameBatchDataset(
        train_game,
        GAMES_PER_EPOCH,
    )

    validation_dataset = ReferentialGameBatchDataset(
        validation_game,
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

    # Build speaker and listener.
    sender = build_sender(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        hidden_size=128,
        object_embedding_dim=16,
    )

    receiver = build_listener(
        vocab_size=VOCAB_SIZE,
        receiver_embed_dim=32,
        receiver_hidden_size=128,
        candidate_embedding_dim=16,
    )

    # Move models to the chosen device.
    sender = sender.to(device)
    receiver = receiver.to(device)

    # Connect the agents using EGG.
    game = core.SenderReceiverRnnGS(
        sender,
        receiver,
        loss_wrapper,
    )

    game = game.to(device)

    optimizer = torch.optim.Adam(
    game.parameters(),
    lr=1e-3,
)

    # EGG training loop.
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=[
            core.ConsoleLogger(
                print_train_loss=True,
                as_json=False,
            ),
        ],
    )

    print("\nStarting training...\n")

    trainer.train(n_epochs=EPOCHS)

    # Save the trained parameters.
    torch.save(
        {
            "model_state_dict": game.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "num_candidates": NUM_CANDIDATES,
            "epochs": EPOCHS,
        },
        "referential_game.pt",
    )

    print("\nTraining complete!")
    print("Model saved to referential_game.pt")


if __name__ == "__main__":
    core.init()
    main()
