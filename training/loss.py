import torch.nn.functional as F


def referential_loss(
    receiver_output,
    labels,
):
    """
    Compute the referential-game loss and accuracy.

    Args:
        receiver_output:
            Candidate scores with shape
            [B, num_candidates].

        labels:
            Correct candidate indices
            with shape [B].

    Returns:
        Per-example cross-entropy loss
        and auxiliary accuracy information.
    """

    loss = F.cross_entropy(
        receiver_output,
        labels,
        reduction="none",
    )

    prediction = receiver_output.argmax(
        dim=1
    )

    accuracy = (
        prediction == labels
    ).float()

    return loss, {
        "accuracy": accuracy,
    }