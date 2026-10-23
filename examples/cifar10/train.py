"""Train a small CNN on CIFAR-10 (60,000 tiny photos, 10 classes: plane, car, bird, cat, ...).

The dataset downloads into data/ on the first run (from Hugging Face) and is reused after that.
Results go to outputs/: the trained model and a metrics.json with accuracy per epoch.
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--out", default="outputs")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--batch-size", type=int, default=256)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def load(split):
    """CIFAR-10 split as uint8 images (N, 3, 32, 32) and labels, cached in data/ after the first run."""
    cache = os.path.join(args.data, f"cifar10-{split}.pt")
    if not os.path.exists(cache):
        import io
        import urllib.request
        import numpy as np
        import pyarrow.parquet as pq
        from PIL import Image
        os.makedirs(args.data, exist_ok=True)
        print(f"downloading CIFAR-10 {split} set...", flush=True)
        raw = urllib.request.urlopen(URL.format(split)).read()
        table = pq.read_table(io.BytesIO(raw)).to_pydict()
        images = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in table["img"]])
        torch.save((torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(), torch.tensor(table["label"])), cache)
    return torch.load(cache)


# the whole dataset fits on the GPU (180 MB), so there's no DataLoader
MEAN = torch.tensor([0.49, 0.48, 0.45], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.24, 0.26], device=device).view(1, 3, 1, 1)
x_train, y_train = (t.to(device) for t in load("train"))
x_test, y_test = (t.to(device) for t in load("test"))
print(f"{len(x_train)} training images, {len(x_test)} test images", flush=True)


def prep(x, augment):
    x = (x.float() / 255 - MEAN) / STD
    if augment:  # random crop (pad 4) and horizontal flip
        x = nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        i, j = torch.randint(0, 9, (2,))
        x = x[:, :, i:i + 32, j:j + 32]
        if torch.rand(1) < 0.5:
            x = x.flip(3)
    return x.contiguous(memory_format=torch.channels_last)


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


model = nn.Sequential(
    block(3, 64), block(64, 128), nn.MaxPool2d(2),
    block(128, 128), block(128, 256), nn.MaxPool2d(2),
    block(256, 256), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
).to(device).to(memory_format=torch.channels_last)

opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
steps_per_epoch = len(x_train) // args.batch_size
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=args.epochs * steps_per_epoch)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
history = []

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(x_train), device=device)
    for k in range(steps_per_epoch):
        idx = perm[k * args.batch_size:(k + 1) * args.batch_size]
        x, y = prep(x_train[idx], augment=True), y_train[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    correct = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(x_test), 1000):
            correct += (model(prep(x_test[k:k + 1000], augment=False)).argmax(1) == y_test[k:k + 1000]).sum().item()
    acc = correct / len(x_test)
    history.append({"epoch": epoch, "loss": round(loss.item(), 4), "test_accuracy": round(acc, 4)})
    print(f"epoch {epoch:2d}/{args.epochs}  loss {loss.item():.3f}  test accuracy {acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)

torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump(history, f, indent=2)
print(f"done: final test accuracy {history[-1]['test_accuracy']:.1%}, model saved to {args.out}/model.pt")
