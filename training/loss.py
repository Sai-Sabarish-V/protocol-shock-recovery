import torch.nn.functional as F


def referential_loss(
    receiver_output,
    labels,
):

    loss = F.cross_entropy(
        receiver_output,
        labels,
        reduction="none",
    )

    prediction = receiver_output.argmax(dim=1)

    accuracy = (prediction == labels).float()

    return loss, {
        "accuracy": accuracy,
    }