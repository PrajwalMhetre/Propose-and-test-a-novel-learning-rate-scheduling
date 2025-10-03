import matplotlib.pyplot as plt
from src.train import train_and_evaluate

epochs = 15
hist_novel = train_and_evaluate("novel", epochs)
hist_step = train_and_evaluate("step", epochs)
hist_cosine = train_and_evaluate("cosine", epochs)

# Plot results
plt.figure(figsize=(12,5))

# Loss curves
plt.subplot(1,2,1)
plt.plot(hist_novel["loss"], label="Novel")
plt.plot(hist_step["loss"], label="StepLR")
plt.plot(hist_cosine["loss"], label="CosineAnnealing")
plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curves
plt.subplot(1,2,2)
plt.plot(hist_novel["val_acc"], label="Novel")
plt.plot(hist_step["val_acc"], label="StepLR")
plt.plot(hist_cosine["val_acc"], label="CosineAnnealing")
plt.title("Validation Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("plots/comparison.png")
plt.show()
