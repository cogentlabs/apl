import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from apl import models
from apl import memory_store
from datasets import omniglot

N_CLASSES = 200
N_NEIGHBOURS = 5
MAX_BATCHES = 3000
MEMORY_SIZE = 10000
SIGMA_RATIO = 0.75
QUERY_EMBED_DIM = 64
LABEL_EMBED_DIM = 32
KEY_SIZE = 256
VALUE_SIZE = 256
N_HEADS = 2
NUM_LAYERS = 5
USE_CUDA = True
SAVE_FREQUENCY = 100

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.long().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot


def split_batch(batch, nshot, n_classes, n_per_class):
    context = []
    query = []
    for i in range(n_classes):
        class_start = i * n_per_class
        context.extend(
            [batch[b] for b in range(class_start, class_start + nshot)])
        query.extend(
            [batch[b] for b in range(class_start + nshot, class_start + n_per_class)])
    return context, query

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--n_classes", type=int, default=N_CLASSES)
    parser.add_argument("--n_neighbours", type=int, default=N_NEIGHBOURS)
    parser.add_argument("--memory_size", type=int, default=MEMORY_SIZE)
    parser.add_argument("--sigma_ratio", type=float, default=SIGMA_RATIO)
    parser.add_argument("--query_embed_dim", type=int, default=QUERY_EMBED_DIM)
    parser.add_argument("--label_embed_dim", type=int, default=LABEL_EMBED_DIM)
    parser.add_argument("--key_size", type=int, default=KEY_SIZE)
    parser.add_argument("--value_size", type=int, default=VALUE_SIZE)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--use_cuda", type=bool, default=USE_CUDA)
    return parser.parse_args()

def test_checkpoint():
    args = get_arguments()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    enc = models.Encoder()
    dec = models.RSAFFDecoder(
        args.n_classes, args.query_embed_dim, args.label_embed_dim,
        args.n_neighbours, args.key_size, args.value_size, args.n_heads,
        args.num_layers)
    enc.to(device)
    dec.to(device)
    memory = memory_store.MemoryStore(
            args.memory_size, args.n_classes,
            args.n_neighbours, args.query_embed_dim, device)
    train_dataset = omniglot.RestrictedOmniglot(
        "data/Omniglot", args.n_classes, train=True, noise_std=0.1)
    test_dataset = omniglot.RestrictedOmniglot(
        "data/Omniglot", args.n_classes, train=False, noise_std=0)
    nll_threshold = args.sigma_ratio * np.log(args.n_classes)

    checkpoint = torch.load(args.checkpoint)
    enc.load_state_dict(checkpoint['encoder_state'])
    dec.load_state_dict(checkpoint['decoder_state'])

    memory.flush()
    enc.eval()
    dec.eval()

    accuracy = []
    ker_accuracy = []
    memory_size = []
    top1_matches = []
    loss_list = []

    # Pick a batch with n_classes classes and 20 examples per class.
    # Test it on the online setting.
    test_dataset.shuffle_classes()
    batch = list(test_dataset)
    shuffled_batch = random.sample(batch, len(batch))

    for batch_idx, (data, target) in enumerate(shuffled_batch):
        target = torch.Tensor([target]).long()
        data, target = data.to(device), target.to(device)
        query_embeds = enc(data.unsqueeze(0))
        buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(query_embeds)
        top1_labels = buffer_labels[:, 0]
        top1_match = float(torch.mean((top1_labels == target).double()))

        logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
        preds = torch.argmax(logprob, dim=1)
        acc = float(torch.mean((preds == target).double()))
        batch_loss = F.cross_entropy(logprob, target, reduce=False)

        n_classes = memory.n_classes
        dist_probs = F.softmax(-distances, dim=1)
        ker_probs = to_one_hot(
            buffer_labels, n_dims=n_classes + 1)[:, :, :n_classes] * dist_probs.unsqueeze(-1)
        ker_probs = torch.sum(ker_probs, dim=1)
        ker_pred = torch.argmax(ker_probs, dim=1)
        ker_acc = float(torch.mean(
            (ker_pred == target).double()))

        surprise_indices = torch.nonzero(batch_loss > nll_threshold)
        for idx in surprise_indices:
            memory.add_entry(query_embeds[idx], target[idx])

        accuracy.append(acc)
        ker_accuracy.append(ker_acc)
        memory_size.append(len(memory))
        top1_matches.append(top1_match)
        loss_list.append(float(torch.mean(batch_loss)))

    accuracy = np.array(accuracy)
    ker_accuracy = np.array(ker_accuracy)
    top1_matches = np.array(top1_matches)

    print("APL (full) / no decoder (kernel) / no decoder (top1)")
    print("Final accuracy (last n_classes items): {:.3f} / {:.3f} / {:.3f}".format(
            np.mean(accuracy[-n_classes:]), np.mean(ker_accuracy[-n_classes:]),
            np.mean(top1_matches[-n_classes:])))
    print("Final avg. memory size {}".format(int(np.mean(memory_size[-n_classes:]))))

    # Now test the same batch but with a fixed context size.
    memory.flush()
    context, query = split_batch(batch, nshot=1, n_classes=n_classes, n_per_class=20)
    for example in context:
        data = example[0].unsqueeze(0)
        target = torch.Tensor([example[1]]).long()
        data, target = data.to(device), target.to(device)
        memory.add_entry(enc(data), target)

    accuracy = []
    ker_accuracy = []
    top1_matches = []
    loss_list = []

    for q in query:
        data, target = q
        data = data.unsqueeze(0)
        target = torch.Tensor([target]).long()
        data, target = data.to(device), target.to(device)

        query_embeds = enc(data)
        buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(query_embeds)
        top1_labels = buffer_labels[:, 0]
        top1_match = float(torch.mean((top1_labels == target).double()))

        logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
        preds = torch.argmax(logprob, dim=1)
        acc = float(torch.mean((preds == target).double()))
        batch_loss = F.cross_entropy(logprob, target, reduce=False)

        n_classes = memory.n_classes
        dist_probs = F.softmax(-distances, dim=1)
        ker_probs = to_one_hot(
            buffer_labels, n_dims=n_classes + 1)[:, :, :n_classes] * dist_probs.unsqueeze(-1)
        ker_probs = torch.sum(ker_probs, dim=1)
        ker_pred = torch.argmax(ker_probs, dim=1)
        ker_acc = float(torch.mean(
            (ker_pred == target).double()))

        accuracy.append(acc)
        ker_accuracy.append(ker_acc)
        top1_matches.append(top1_match)
        loss_list.append(float(torch.mean(batch_loss)))

    print("APL (full) / no decoder (kernel) / no decoder (top1)")
    print("Avg. accuracy: {:.3f} / {:.3f} / {:.3f}".format(
            np.mean(accuracy), np.mean(ker_accuracy), np.mean(top1_matches)))

if __name__ == "__main__":
    test_checkpoint()
