import math

class NovelLRScheduler:
    def __init__(self, optimizer, base_lr=0.01, alpha=0.02, beta=0.3, gamma=0.5, warmup_epochs=5):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def step(self):
        if self.epoch < self.warmup_epochs:
            lr = self.base_lr * (self.epoch + 1) / self.warmup_epochs
        else:
            t = self.epoch - self.warmup_epochs
            lr = self.base_lr * math.exp(-self.alpha * t) * (1 + self.beta * math.sin(self.gamma * t))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.epoch += 1
        return lr
