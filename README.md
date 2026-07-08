# Fashion-MNIST Classification with a Feedforward ANN (PyTorch)

A simple feedforward Artificial Neural Network built in PyTorch to classify clothing images from the Fashion-MNIST dataset into 10 categories.

## Overview

This project trains a fully-connected (dense) neural network — no convolutions — to classify 28x28 grayscale images of clothing items. The images are flattened into 784-length vectors and passed through two hidden layers before a final classification layer.

## Dataset

- **Source:** Fashion-MNIST, loaded via `torchvision.datasets.FashionMNIST` (auto-downloaded)
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Split:**
  - 55,000 training images
  - 5,000 validation images (held out from the training set)
  - 10,000 test images

## Preprocessing

- Pixel values normalized from the `0–255` range to `0–1`
- Images flattened from `28x28` to a `784`-length vector inside the model (via `nn.Flatten()`)

## Model Architecture

```
Input (784)
  → Linear(784, 200) → ReLU
  → Linear(200, 100) → ReLU
  → Linear(100, 10)   # raw logits, CrossEntropyLoss applies softmax internally
```

Defined in PyTorch as:

```python
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputlayer = nn.Flatten()
        self.hidden1 = nn.Linear(784, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = self.inputlayer(x)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x
```

## Training Details

| Setting | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam (default learning rate) |
| Epochs | 30 |
| Batch size | 32 |
| Device | CUDA if available, else CPU |

Training and validation loss/accuracy are tracked each epoch and stored in a `history` dict for plotting.

## Model Summary

```
Layer (type:depth-idx)                  Param #
================================================
ANN                                     --
├─Flatten: 1-1                          --
├─Linear: 1-2                           157,000
├─Linear: 1-3                           20,100
├─Linear: 1-4                           1,010
================================================
Total params: 178,110
Trainable params: 178,110
Non-trainable params: 0
```

## Evaluation

- Final test set loss and accuracy are computed after training
- Training history (loss & accuracy, train vs. validation) is plotted across epochs
- A few sample test images are visualized with their **predicted vs. actual** labels, marked ✓ or ✗

### Training Curve Observations

The model was trained for 30 epochs. Training accuracy climbed steadily to over 95%, but validation accuracy plateaued around epoch 10-11 (~89-90%) and validation loss began *increasing* from there even as training loss kept falling — a textbook sign of **overfitting**. This is expected for a plain dense network with no regularization (no Dropout, no BatchNorm, no weight decay) and is a natural jumping-off point for the improvements listed below.

## Model Saving / Loading

The trained model's weights are saved and can be reloaded for inference:

```python
torch.save(model.state_dict(), 'fashion_ann_pytorch.pth')

loaded_model = ANN().to(device)
loaded_model.load_state_dict(torch.load('fashion_ann_pytorch.pth', weights_only=True))
loaded_model.eval()
```

## Tech Stack

- Python
- PyTorch, torchvision
- NumPy, pandas
- matplotlib, seaborn
- torchinfo (for model summary)

## How to Run

```bash
pip install torch torchvision numpy pandas matplotlib seaborn torchinfo

jupyter notebook FashionMnsit.ipynb
```

Run all cells in order — the dataset downloads automatically on first run.

## Results

| Metric | Value |
|---|---|
| Final training accuracy (epoch 30) | 95.07% |
| Final validation accuracy (epoch 30) | 89.50% |
| Best validation accuracy (epoch 20) | 90.16% |
| **Test accuracy** | **89.08%** |
| **Test loss** | **0.4646** |

The gap between training accuracy (~95%) and test accuracy (~89%) reflects the overfitting noted above — the model memorizes training examples more than it generalizes, which regularization techniques (Dropout, weight decay, early stopping) would help close.




