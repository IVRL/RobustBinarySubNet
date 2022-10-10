import torch

class AverageCalculator(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0.
        self.average = 0.

    def update(self, value, weight = 1):
        self.value = value
        self.sum += value * weight
        self.count += weight
        self.average = self.sum / self.count


def accuracy2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
