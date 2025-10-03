import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.model import SimpleCNN
from src.scheduler import NovelLRScheduler

# Dataset
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Training Loop
def train_and_evaluate(scheduler_type="novel", epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Scheduler selection
    if scheduler_type == "novel":
        scheduler = NovelLRScheduler(optimizer)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader, _ = get_dataloaders()

    history = {"loss": [], "val_acc": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        val_acc = correct / total

        # Scheduler step
        if scheduler_type == "novel":
            lr = scheduler.step()
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

        history["loss"].append(total_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(f"[{scheduler_type}] Epoch {epoch+1}/{epochs} | Loss={total_loss/len(train_loader):.4f} | Val Acc={val_acc:.4f} | LR={lr:.6f}")

    return history
