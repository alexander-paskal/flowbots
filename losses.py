"""
contains loss and metric functions
"""

import torch


def epe_loss(pred, label):
    """
    Defines expected predicted error loss using p2 norm
    :param pred:
    :param label:
    :return:
    """
    loss = torch.norm(label - pred, p=2, dim=1).mean()
    return loss


def f1_all(pred, label):
    """
    Returns the f1_all error
    :param pred:
    :param label:
    :return:
    """
    pred = torch.norm(pred, p=2, dim=1)
    label = torch.norm(label, p=2, dim=1)

    diff = pred - label
    diff_percent = torch.divide(label, diff)  # calculates the difference in the metrics

    percent_error = diff_percent >= 0.05
    # pixel_error = diff >= 3

    failure = torch.logical_and(percent_error, percent_error)
    failure = torch.flatten(failure)

    ratios = torch.divide(torch.where(failure)[0].size(0), failure.size(0))
    batch_ratio = ratios.mean()
    return batch_ratio

