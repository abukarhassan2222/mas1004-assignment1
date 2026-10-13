"""Build a model and train it. YOU write the bodies of these functions.

The model is ResNet18, a convolutional network that was already trained on
ImageNet, 1.2 million photographs of 1,000 kinds of thing. You keep everything
it learned there and replace only its last layer, so that it answers with your
categories instead of those 1,000. Then you train it further on your images.

Run `pytest tests/test_train.py` after you fill them in. The first run
downloads the ImageNet weights, about 45 MB.
"""

import torch
import torch.nn as nn


def build_model(num_classes, pretrained=True, freeze=False):
    """Return a ResNet18 whose last layer has `num_classes` outputs.

    Arguments
        num_classes  int, how many categories you have
        pretrained   True: start from the ImageNet weights,
                     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
                     False: start from random numbers, as if ImageNet had
                     never happened. You need this for the comparison in
                     Problem 3.
        freeze       True: only the new last layer is trained. Every other
                     parameter keeps the value it came with, which you do by
                     setting requires_grad to False on it.
                     False: every parameter is trained.

    Returns the model from torchvision.models.resnet18, with its `fc`
    attribute replaced by a new nn.Linear that has 512 inputs and num_classes
    outputs.

    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # The fresh last layer starts with requires_grad True, so with freeze=True
    # it is the only thing training is allowed to change.
    model.fc = nn.Linear(512, num_classes)
    return model


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, lr=1e-4, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned, or any other torch model
        X_train      np.float32 (N, ...)      y_train  np.int64 (N,)
        X_test       np.float32 (M, ...)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  the mean loss over the batches of that epoch
        "train_acc"   the share of training rows the model got right in those
                      batches, between 0.0 and 1.0
        "test_loss"   the mean loss over the test set, measured after the epoch
        "test_acc"    the share of the test set it gets right, measured after
                      the epoch, between 0.0 and 1.0

    What to do
        Use a GPU when there is one:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Move the model there once, and each batch there as you use it. Keep
        X_train and X_test themselves in ordinary memory: they may not fit on
        the GPU.
        Shuffle the training rows again at the start of every epoch.
        Use nn.CrossEntropyLoss, which expects raw outputs, and torch.optim.Adam
        over the parameters that have requires_grad set.
        Call model.train() before the training batches and model.eval() before
        measuring the test set. ResNet contains BatchNorm layers, which behave
        differently in the two modes.
        Do not compute gradients while measuring the test set, and never let
        the test rows change the parameters.
        Print one line per epoch, so that you can watch it while it runs.

    The model is trained in place. When this function returns, `model` is the
    trained one, and it is left on the device it was trained on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    criterion = nn.CrossEntropyLoss()
    trainable = (p for p in model.parameters() if p.requires_grad)
    optimiser = torch.optim.Adam(trainable, lr=lr)

    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}
    n = len(X_train_t)

    for epoch in range(1, epochs + 1):
        # Shuffle the training rows again at the start of every epoch.
        perm = torch.randperm(n)
        X_ep = X_train_t[perm]
        y_ep = y_train_t[perm]

        model.train()
        loss_sum, correct, steps = 0.0, 0, 0
        for start in range(0, n, batch_size):
            xb = X_ep[start:start + batch_size].to(device)
            yb = y_ep[start:start + batch_size].to(device)

            optimiser.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimiser.step()

            loss_sum += float(loss.item())
            correct += int((out.argmax(1) == yb).sum().item())
            steps += 1

        train_loss = loss_sum / steps
        train_acc = correct / n

        # Measure the test set without changing anything.
        model.eval()
        with torch.no_grad():
            out = model(X_test_t.to(device))
            test_loss = float(criterion(out, y_test_t.to(device)).item())
            test_acc = float((out.argmax(1) == y_test_t.to(device)).sum().item()) / len(y_test_t)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:3d}/{epochs}: train_loss {train_loss:.4f} "
              f"train_acc {train_acc:.3f}  test_loss {test_loss:.4f} "
              f"test_acc {test_acc:.3f}")

    return history


def count_trainable(model):
    """How many parameters training is allowed to change. Written for you."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, path):
    """Save a trained model. Written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`, onto the CPU. Written for you."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
