import egg.core as core
import torch.nn as nn

import torch
import torch.nn as nn

class SpeakerAgent(nn.Module):
    def __init__(self, input_dim=5, hidden_size=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, aux_input=None):
        return self.encoder(x)

from egg.core import RnnSenderGS

def build_sender(
    input_dim=5,
    hidden_size=128,
    embed_dim=32,
    vocab_size=12,
):
    agent = SpeakerAgent(
        input_dim=input_dim,
        hidden_size=hidden_size,
    )

    sender = RnnSenderGS(
        agent=agent,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        max_len=3,
        cell="gru"
    )

    return sender

speaker_agent = SpeakerAgent(
    input_dim=5,
    hidden_size=128
)

sender = RnnSenderGS(
    agent=speaker_agent,
    vocab_size=10,
    embed_dim=32,
    hidden_size=128,
    max_len=3,
    temperature=1.0,
    cell="gru"
)


