import copy
import torch


BUFFER_SIZE = 4


class SnapshotBuffer:
    def __init__(self, model, buffer_size=BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.snapshots = []
        self.push(model)

    def push(self, model):
        state = copy.deepcopy(model.state_dict())
        self.snapshots.append(state)
        if len(self.snapshots) > self.buffer_size:
            self.snapshots.pop(0)

    def compute_distance_loss(self, model):
        current = {k: v.float() for k, v in model.state_dict().items()}
        losses = []
        for snapshot in self.snapshots:
            for k in current:
                if current[k].shape == snapshot[k].shape:
                    diff = (
                        current[k] - snapshot[k].to(current[k].device).float()
                    ).norm()
                    losses.append(-diff)
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
