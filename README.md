# Protocol Shock Recovery

## Objective

This project investigates how an emergent communication protocol adapts after a sudden reduction in communication bandwidth.

Using the EGG (Emergence of Grounded Language Games) framework, we train two neural agents (Speaker and Listener) to communicate through a discrete communication channel. After convergence, the communication bandwidth is reduced (e.g., message length or vocabulary size), and we study how the protocol reorganizes to recover performance.

---

## Research Questions

- How does communication change after bandwidth reduction?
- Does the protocol recover completely?
- Which information is preserved first?
- How efficiently can agents reorganize their communication?

---

## Framework

- Python
- PyTorch
- Facebook Research EGG
- Gumbel-Softmax Communication

---

# Repository Structure

## `configs/`

Contains all experiment hyperparameters.

Examples:

- Vocabulary size
- Message length
- Hidden dimension
- Learning rate
- Batch size

Instead of changing values throughout the codebase, all configurable settings should be stored here.

---

## `data/`

Responsible for dataset generation.

Contains code for:

- Synthetic object generation
- Referential game creation
- Data loading
- Utility functions

Example object:

```
Color = Red
Shape = Circle
Size = Small
Material = Wood
Pattern = Striped
```

This folder should not contain any neural network code.

---

## `models/`

Contains all neural network architectures.

Speaker:
- Receives an object
- Produces the initial hidden representation

Listener:
- Receives the decoded message representation
- Predicts the target object (or reconstructs features)

Only model definitions belong here.

---

## `training/`

Contains the training pipeline.

Responsibilities:

- Build the EGG game
- Initialize Speaker and Listener
- Configure optimizer
- Run training
- Compute losses
- Apply bandwidth reduction experiments

This folder controls learning but does not define datasets or models.

---

## `analysis/`

Contains scripts used after training.

Examples:

- Accuracy plots
- Vocabulary usage
- Message statistics
- Protocol entropy
- Recovery curves
- Communication efficiency

Everything related to analyzing trained models belongs here.

---

## `experiments/`

Stores experiment-specific configurations.

Examples:

```
baseline/

bandwidth_3/

bandwidth_2/

bandwidth_1/
```

Each experiment should have its own configuration and outputs to make experiments reproducible.

---

## `checkpoints/`

Stores trained model weights.

Examples:

```
speaker_epoch100.pt

listener_epoch100.pt
```

These files are generated during training.

---

## `outputs/`

Stores generated outputs.

Examples:

- Graphs
- Logs
- Generated messages
- Evaluation results

These files are automatically created and are not committed to Git.

---

## `docs/`

Project documentation.

Contains:

- Research notes
- Meeting notes
- Design decisions
- Experiment ideas
- Future work

Think of this folder as the project's research notebook.

---

# High-Level Pipeline

```
Synthetic Dataset
        │
        ▼
Speaker Network
        │
        ▼
RnnSenderGS (EGG)
        │
        ▼
Discrete Communication Message
        │
        ▼
RnnReceiverGS (EGG)
        │
        ▼
Listener Network
        │
        ▼
Prediction
        │
        ▼
Loss
        │
        ▼
Backpropagation
```

---
