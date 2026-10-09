
import torch
import torch.nn as nn

from egg.core import RnnReceiverGS


NUM_FEATURES = 5
NUM_FEATURE_VALUES = 4


class CandidateEncoder(nn.Module):

    def __init__(
        self,
        num_values=NUM_FEATURE_VALUES,
        embedding_dim=16,
    ):
        super().__init__()

        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(num_values, embedding_dim)
            for _ in range(NUM_FEATURES)
        ])

    def forward(self, candidates):
        """
        candidates: [B, num_candidates, 5]
        returns:    [B, num_candidates, 80]
        """
        candidates = candidates.long()
        embeddings = []

        for feature_idx in range(NUM_FEATURES):
            feature_values = candidates[:, :, feature_idx]

            feature_embedding = self.feature_embeddings[
                feature_idx
            ](feature_values)

            embeddings.append(feature_embedding)

        return torch.cat(embeddings, dim=-1)


class ListenerAgent(nn.Module):

    def __init__(
        self,
        receiver_hidden_size=128,
        candidate_embedding_dim=16,
    ):
        super().__init__()

        self.candidate_encoder = CandidateEncoder(
            num_values=NUM_FEATURE_VALUES,
            embedding_dim=candidate_embedding_dim,
        )

        candidate_vector_size = (
            NUM_FEATURES * candidate_embedding_dim
        )

        self.message_projection = nn.Linear(
            receiver_hidden_size,
            candidate_vector_size,
        )

    # IMPORTANT: forward must be indented inside ListenerAgent
    def forward(
        self,
        hidden_state,
        input=None,
        aux_input=None,
    ):
        """
        hidden_state: [B, receiver_hidden_size]
        input:        [B, num_candidates, 5]
        returns:      [B, num_candidates]
        """
        if input is None:
            raise ValueError(
                "Listener requires candidates as receiver input."
            )

        candidates = input.long()

        candidate_vectors = self.candidate_encoder(candidates)

        message_vector = self.message_projection(
            hidden_state
        ).unsqueeze(1)

        scores = (
            candidate_vectors * message_vector
        ).sum(dim=-1)

        return scores


def build_listener(
    vocab_size=12,
    receiver_embed_dim=32,
    receiver_hidden_size=128,
    candidate_embedding_dim=16,
):
    agent = ListenerAgent(
        receiver_hidden_size=receiver_hidden_size,
        candidate_embedding_dim=candidate_embedding_dim,
    )

    receiver = RnnReceiverGS(
        agent=agent,
        vocab_size=vocab_size,
        embed_dim=receiver_embed_dim,
        hidden_size=receiver_hidden_size,
        cell="gru",
    )

    return receiver
