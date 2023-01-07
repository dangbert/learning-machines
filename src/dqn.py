#!/usr/bin/env python3
import torch
from torch import nn


class DQN(nn.Module):
    stack: nn.Sequential

    def __init__(self, input_len: int, output_len: int):
        """Note for LM project, there are 8 IRs so input_len could be 8"""
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(input_len, 30),
            nn.Sigmoid(),
            nn.Linear(30, 15),
            nn.Sigmoid(),
            nn.Linear(15, output_len),
            nn.Softmax(dim=output_len),
        )

    def forward(self, x):
        """
        Defines the forward computation at every call.
        Note: This shouldn't be called directly.
        """
        # TODO: perhaps normalize IR inputs first (remove -infinity?
        logits = self.stack(x)
        return logits


def main():
    pass


if __name__ == "__main__":
    main()
