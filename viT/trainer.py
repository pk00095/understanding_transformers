import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from encoder import VisionTransformer
from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch, logging_steps):
    model.train()
    loss_func = nn.CrossEntropyLoss()

    epoch_loss = 0
    batches_seen = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batches_seen += data.shape[0]
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % logging_steps == 0:
            # print(data.shape)
            # print(f"target shape {target.shape}, pred shape {output.shape}" )
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), epoch_loss/batches_seen))
            # if args.dry_run:
            #     break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    pbar = tqdm(total=len(test_loader.dataset))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.update(data.shape[0])
    pbar.close()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

def predict(model, x):
    res = model(x)
    return res.argmax(dim=1)

if __name__ == "__main__":
    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 2}

    device = "cpu"

    batch_size = 32
    height = 28
    width = 28
    channels = 1
    num_classes = 10

    d_model = 64
    patch_size = 4
    num_heads = 8
    num_layers = 3

    epochs = 10
    logging_steps = 50


    transform=transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = VisionTransformer(
        image_dims=(height, width, channels),
        n_layers=num_layers,
        patch_size=patch_size,
        d_model=d_model,
        d_intermediate=d_model*2,
        num_heads=num_heads,
        num_classes=num_classes
    )

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, logging_steps)
        test(model, device, test_loader)
        torch.save(model.state_dict(), f"mnist_viT-epoch_{epoch}.pt")
        scheduler.step()