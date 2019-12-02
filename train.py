def train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.
    for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
        optimizer.zero_grad()
        anc_embedding = model(anchor)
        pos_embedding = model(positive)
        neg_embedding = model(negative)
        loss = criterion(anc_embedding, pos_embedding, neg_embedding)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 5 == 4:
            print(f'epoch{epoch}, batch{batch_idx+1} loss: {running_loss / 5}')
            train_loss = running_loss / 5
            running_loss = 0.

    return train_loss
