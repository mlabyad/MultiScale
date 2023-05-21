import torch
import torch.nn.functional as F

def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()

def save_checkpoint(model, optimizer, preconditioner, schedulers, filepath):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'preconditioner': preconditioner.state_dict()
                if preconditioner is not None else None,
        'schedulers': [s.state_dict() for s in schedulers]
                if isinstance(schedulers, list) else None
    }
    torch.save(state, filepath)

class LabelSmoothLoss(torch.nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


