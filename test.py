import torch


def test(args, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for anchor, positive, negative, anchor_label in test_loader:
            anc_embedding = model(anchor)
            pos_embedding = model(positive)
            neg_embedding = model(negative)
            test_loss += criterion().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    num_dataset = len(test_loader.dataset)
    test_loss /= num_dataset
    accuracy = correct / num_dataset

    print(f'\nTest set:')
    print(f'Average loss: {test_loss}')
    print(f'Accuracy: {correct} / {num_dataset} ({accuracy*100})\n')

    return test_loss, accuracy
