import torch


def accuracy(predictions, labels):
    """
    Calculate the accuracy of predictions against true labels.
    """
    with torch.no_grad():
        return (predictions.squeeze()==labels.squeeze()).float().mean()
