import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from apl import models
from apl import memory_store
from datasets import omniglot

N_CLASSES = 200
N_NEIGHBOURS = 5
N_EPISODES = 20000
TEST_FREQUENCY = 100
BATCH_SIZE = 32
EPOCH_PER_EPISODE = 10
MAX_BATCHES = 3000
MEMORY_SIZE = 10000
SIGMA_RATIO = 0.75
DECODER_TYPE = "RSAFF"
QUERY_EMBED_DIM = 64
LABEL_EMBED_DIM = 32
KEY_SIZE = 256
VALUE_SIZE = 256
N_HEADS = 2
NUM_LAYERS = 5
LR = 0.0001
USE_CUDA = True
SAVE_FREQUENCY = 100
ENC_CKPT = None


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_checkpoint", type=str, default=ENC_CKPT)
    parser.add_argument("--n_classes", type=int, default=N_CLASSES)
    parser.add_argument("--n_neighbours", type=int, default=N_NEIGHBOURS)
    parser.add_argument("--n_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--test_frequency", type=int, default=TEST_FREQUENCY)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--memory_size", type=int, default=MEMORY_SIZE)
    parser.add_argument("--sigma_ratio", type=float, default=SIGMA_RATIO)
    parser.add_argument("--decoder_type", type=str, default=DECODER_TYPE)
    parser.add_argument("--query_embed_dim", type=int, default=QUERY_EMBED_DIM)
    parser.add_argument("--label_embed_dim", type=int, default=LABEL_EMBED_DIM)
    parser.add_argument("--key_size", type=int, default=KEY_SIZE)
    parser.add_argument("--value_size", type=int, default=VALUE_SIZE)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--use_cuda", type=bool, default=USE_CUDA)
    parser.add_argument("--save_frequency", type=int, default=SAVE_FREQUENCY)
    return parser.parse_args()


def apl_train(enc, dec, memory, dataset, device, batch_size, max_batches, nll_threshold, opt):
    enc.train()
    dec.train()

    dataset.shuffle_classes()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    memory.flush()

    # Variables to record the length of next episode
    next_max_batches = 2 * max_batches
    seen_perfect_acc = False
    accuracy_list = []
    nll_list = []

    for batch_idx in range(int(max_batches)):
        data_gen = enumerate(loader)
        _, (data, target) = next(data_gen)
        data, target = data.to(device), target.to(device)

        opt.zero_grad()

        query_embeds = enc(data)
        buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(
            query_embeds.detach())

        logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
        preds = torch.argmax(logprob, dim=1)
        acc = float(torch.mean((preds == target).double()))
        batch_loss = F.cross_entropy(logprob, target, reduce=False)
        target_loss = torch.mean(batch_loss)

        if np.random.rand() > 0.5:
            target_loss.backward()
            nn.utils.clip_grad_value_(enc.parameters(), 1)
            nn.utils.clip_grad_value_(dec.parameters(), 1)
            opt.step()

        surprise_indices = torch.nonzero(batch_loss > nll_threshold)
        if surprise_indices.size()[0] > 0:
            surprise_indices = surprise_indices.squeeze(1)
            memory.add_batched_entries(
                query_embeds[surprise_indices].detach(), target[surprise_indices].detach())

        # Capping number of next iterations if model currently performs well
        if acc == 1.0 and not seen_perfect_acc:
            next_max_batches = 3 * (batch_idx + 1)
            seen_perfect_acc = True

        accuracy_list.append(float(acc))
        nll_list.append(float(target_loss))

    return accuracy_list, nll_list, len(memory), next_max_batches

def apl_test(enc, dec, memory, dataset, device, batch_size, max_batches, nll_threshold):
    enc.eval()
    dec.eval()

    dataset.shuffle_classes()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)
    memory.flush()

    accuracy_list = []
    nll_list = []

    for batch_idx in range(int(max_batches)):
        data_gen = enumerate(loader)
        _, (data, target) = next(data_gen)
        data, target = data.to(device), target.to(device)

        query_embeds = enc(data)
        buffer_embeds, buffer_labels, distances = memory.get_nearest_entries(
            query_embeds.detach())

        logprob = dec(buffer_embeds, buffer_labels, query_embeds, distances)
        preds = torch.argmax(logprob, dim=1)
        acc = float(torch.mean((preds == target).double()))
        batch_loss = F.cross_entropy(logprob, target, reduce=False)
        target_loss = torch.mean(batch_loss)

        surprise_indices = torch.nonzero(batch_loss > nll_threshold)
        if surprise_indices.size()[0] > 0:
            surprise_indices = surprise_indices.squeeze(1)
            memory.add_batched_entries(
                query_embeds[surprise_indices].detach(), target[surprise_indices].detach())

        accuracy_list.append(float(acc))
        nll_list.append(float(target_loss))

    return accuracy_list, nll_list, len(memory)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M", time.localtime())


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def run_omniglot():
    timestamp = get_time()
    args = get_arguments()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    enc = models.Encoder().to(device)
    enc.apply(weights_init)
    dec = models.RSAFFDecoder(
            args.n_classes, args.query_embed_dim, args.label_embed_dim,
            args.n_neighbours, args.key_size, args.value_size, args.n_heads,
            args.num_layers).to(device)
    dec.apply(weights_init)
    memory = memory_store.MemoryStore(args.memory_size, args.n_classes,
                         args.n_neighbours, args.query_embed_dim, device)
    train_dataset = omniglot.RestrictedOmniglot(
        "data/Omniglot", args.n_classes, train=True, noise_std=0.1)
    test_dataset = omniglot.RestrictedOmniglot(
        "data/Omniglot", args.n_classes, train=False, noise_std=0)
    nll_threshold = args.sigma_ratio * np.log(args.n_classes)

    max_batches = args.max_batches
    print(enc)
    print(dec)
    if args.encoder_checkpoint:
        opt = optim.Adam(list(dec.parameters()), lr=args.lr)
        print('loading encoder checkpoint from {}'.format(args.encoder_checkpoint))
        checkpoint = torch.load(args.encoder_checkpoint)
        enc.load_state_dict(checkpoint['encoder_state'])
    else:
        print('training encoder from scratch!!')
        opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                         lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        opt, [1000, 2000, 5000, 10000], gamma=0.5)

    for episode_idx in range(args.n_episodes):
        start_time = time.time()
        acc, nll, used_memory, next_batches = apl_train(
            enc, dec, memory, train_dataset, device, args.batch_size,
            max_batches, nll_threshold, opt)
        used_time = time.time() - start_time
        max_batches = min(args.max_batches, next_batches)
        print("{} acc:{:.3f} nll:{:.3f} b:{} mem:{} [{:.1f} s/it]".format(
            episode_idx, np.mean(acc), np.mean(nll), max_batches, used_memory, used_time))

        scheduler.step()

        if episode_idx % args.test_frequency == 0 and episode_idx != 0:
            acc, nll, used_memory = apl_test(
                enc, dec, memory, test_dataset, device, 1, 8 * args.n_classes,
                nll_threshold)
            print("Test -> acc:{:.3f} nll:{:.3f} mem:{} [{:.1f} s/it]".format(
                np.mean(acc), np.mean(nll), used_memory, used_time))

        if episode_idx % args.save_frequency == 0 or (episode_idx == (args.n_episodes - 1)):
            directory = 'data/checkpoints/{}_{}_{}'.format(
                args.decoder_type, args.n_classes, timestamp)
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = directory + '/ckpt_{}.pth'.format(episode_idx)
            print('saving on ', save_path)
            torch.save({
                'encoder_state': enc.state_dict(),
                'decoder_state': dec.state_dict(),
            }, save_path)


if __name__ == "__main__":
    run_omniglot()
