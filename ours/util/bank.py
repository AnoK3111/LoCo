import torch
import torch.nn as nn

class Bank(nn.Module):
    def __init__(self, num_classes, lam=0.0, warmup_step=1000, device='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.lam = lam
        self.warmup_step = warmup_step
        self.device = device
        self.register_buffer('prototype', torch.zeros(num_classes, 64).to(device))
        self.step = 0

    def update(self, feature, label):
        feature = feature.detach()
        unique_cls = torch.unique(label)
        for c in unique_cls:
            c_feature = feature[label == c].reshape(-1, 64)
            new_feature = c_feature.mean(dim=0)
            if  (self.prototype[c] == 0).all() or self.step <= self.warmup_step:
                self.prototype[c] = new_feature
            else:
                self.prototype[c] = self.lam * self.prototype[c] + (1 - self.lam) * new_feature
        self.step += 1

    def get_prototype(self):
        prototype_feature = self.prototype.clone()
        feature = []
        label = []
        for c in range(self.num_classes):
            if (self.prototype[c] == 0).all():
                continue
            feature.append(prototype_feature[c].unsqueeze(0))
            label.append(c)
        feature = torch.cat(feature, dim=0)
        label = torch.tensor(label).to(self.device)

        return feature, label
    
if __name__=="__main__":
    b = Bank(3)
    x = torch.rand(4, 64)
    y0 = torch.tensor([0, 0, 1, 1])
    y1 = torch.tensor([0, 0, 1, 2])
    b.update(x, y0)
    print(b.get_prototype())
    b.update(x, y1)
    print(b.get_prototype())
