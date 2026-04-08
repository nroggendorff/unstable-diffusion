import torch


BUFFER_SIZE = 4


class SnapshotBuffer:
    def __init__(self, model, buffer_size=BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.snapshots = []
        self.push(model)

    def push(self, model):
        state = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.snapshots.append(state)
        if len(self.snapshots) > self.buffer_size:
            self.snapshots.pop(0)

    def compute_distance_loss(self, model):
        losses = []
        params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        for snapshot in self.snapshots:
            for name, param in params.items():
                if name in snapshot and snapshot[name].shape == param.shape:
                    ref = snapshot[name].to(param.device).float()
                    diff = (param.float() - ref).norm()
                    losses.append(-diff)
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
