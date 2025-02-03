import torch


def train(model, X, y):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9, patience=100)
    # criterion = nn.SmoothL1Loss(beta=0.3)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_state_dict = None
    EPOCHS = 200001
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(X)

        # Compute the loss and its gradients
        loss = criterion(outputs, y)
        loss.backward()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state_dict = model.state_dict()

        # Adjust learning weights
        optimizer.step()
        # scheduler.step(loss)
        if epoch % 10000 == 0:
            print(f"Epoch: {epoch + 1}, loss: {loss.item()}")

    # print(f"Last learning rate: {scheduler.get_last_lr()}")
    # Load the best model state dict
    model.load_state_dict(best_state_dict)
