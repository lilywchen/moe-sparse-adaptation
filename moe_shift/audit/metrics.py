"""Classification accuracy."""
import torch


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y, _site, *_ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
