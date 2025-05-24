import torch

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    model.train()
    model.to(device)

    total_loss, total_acc = 0.0, 0.0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
        total_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader), total_acc / len(data_loader)


def test_step(model, data_loader, loss_fn, accuracy_fn, device):
    model.eval()
    model.to(device)

    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    return total_loss / len(data_loader), total_acc / len(data_loader)
