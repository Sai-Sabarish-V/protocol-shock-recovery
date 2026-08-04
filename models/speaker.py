import torch
import torch.nn as nn

from egg.core import RnnSenderGS


class ObjectEncoder(nn.Module):
    def __init__(self, num_values=4, embedding_dim=16):
        super().__init__()

        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(num_values, embedding_dim)
            for _ in range(5)
        ])

    def forward(self, objects):
        """
        objects:
            [batch_size, 5]

        returns:
            [batch_size, 5 * embedding_dim]
        """

        # Embedding layers require integer indices
        objects = objects.long()

        embeddings = []

        for feature_idx in range(5):
            feature_values = objects[:, feature_idx]

            feature_embedding = self.feature_embeddings[feature_idx](
                feature_values
            )

            embeddings.append(feature_embedding)

        object_vector = torch.cat(embeddings, dim=-1)

        return object_vector


class SpeakerAgent(nn.Module):
    def __init__(
        self,
        num_values=4,
        embedding_dim=16,
        hidden_size=128,
    ):
        super().__init__()

        self.object_encoder = ObjectEncoder(
            num_values=num_values,
            embedding_dim=embedding_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(5 * embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, objects, aux_input=None):
        """
        objects:
            [batch_size, 5]
        returns:
            [batch_size, hidden_size]
        """

        object_vector = self.object_encoder(objects)
        hidden = self.projection(object_vector)

        return hidden


def build_sender(
    vocab_size=12,
    embed_dim=32,
    hidden_size=128,
    object_embedding_dim=16,
):
    agent = SpeakerAgent(
        num_values=4,
        embedding_dim=object_embedding_dim,
        hidden_size=hidden_size,
    )

    sender = RnnSenderGS(
        agent=agent,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        max_len=3,
        cell="gru",
        temperature=1.0,
    )

    return sender
