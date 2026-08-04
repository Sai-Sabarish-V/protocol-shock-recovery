import random

import torch



class ReferentialGame:
    def __init__(self, dataset, num_candidates=8):
        self.dataset = dataset
        self.num_candidates = num_candidates

    def sample_game(self):
        indices = random.sample(
            range(len(self.dataset)),
            self.num_candidates,
        )

        target_dataset_index = indices[0]

        random.shuffle(indices)

        target_index = indices.index(
            target_dataset_index
        )

        candidates = torch.stack([
            self.dataset[i]["features"]
            for i in indices
        ])

        target = self.dataset[
            target_dataset_index
        ]["features"]

        return {
            "target": target,
            "candidates": candidates,
            "target_index": target_index,
        }

    def sample_batch(self, batch_size):
        games = [
            self.sample_game()
            for _ in range(batch_size)
        ]

        targets = torch.stack([
            game["target"]
            for game in games
        ])

        candidates = torch.stack([
            game["candidates"]
            for game in games
        ])

        target_indices = torch.tensor(
            [
                game["target_index"]
                for game in games
            ],
            dtype=torch.long,
        )

        return {
            "target": targets,
            "candidates": candidates,
            "target_index": target_indices,
        }