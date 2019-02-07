import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHDPA(nn.Module):
    """Multi-Head Dot-Product attention as defined in https://arxiv.org/pdf/1806.01822.pdf"""

    def __init__(self, memory_slots, key_size, value_size, n_heads):
        """
        Args:
            memory_slots: the number of entries in the memory
            key_size: the dimensionality of keys and queries (2nd dim in Q, K matrices)
            value_size: the dimensionality of values (2nd dim in V matrix)
            n_heads: number of separate DPA blocks
            extra_input: a boolean flag to indicate if extra input given to memory before self-attention

        """
        super(MHDPA, self).__init__()
        self.memory_slots = memory_slots
        self.key_size = key_size
        self.value_size = value_size
        self.n_heads = n_heads

        self.memory_size = self.value_size * self.n_heads
        self.projection_size = self.n_heads * \
            (2 * self.key_size + self.value_size)
        self.qkv_projector = nn.Linear(self.memory_size, self.projection_size)
        self.qkv_layernorm = nn.LayerNorm(
            [self.memory_slots, self.projection_size])
        self.qkv_size = 2 * self.key_size + self.value_size

    def forward(self, memory):
        """
        https://github.com/L0SG/relational-rnn-pytorch/blob/master/relational_rnn_general.py as main reference
        In the code below:
        B - batch size
        N - number of memory slots (= number of nearest neighbours)
        V - self.value_size in single DPA
        K - self.key_size in single DPA
        H - number of heads in MHDPA
        M - memory size, M = V * H
        Args:
            memory: memory tensor of (B, N, M) shape
            memory_input: extra entry to add to memory in the beginning with (B, M) shape
        Return:
            new memory tensor with (B, N, M) shape

        """
        # First, a simple linear projection is used to construct queries
        qkv = self.qkv_projector(memory)
        # apply layernorm for every dim except the batch dim
        qkv = self.qkv_layernorm(qkv)

        # mem_slots needs to be dynamically computed since mem_slots got
        # concatenated with inputs. example: self.mem_slots=10 and seq_length
        # is 3, and then mem_slots is 10 + 1 = 11 for each 3 step forward pass
        # this is the same as self.mem_slots_plus_input, but defined to keep
        # the sonnet implementation code style
        mem_slots = memory.shape[1]  # denoted as N

        # split the qkv to multiple heads H
        # [B, N, F] => [B, N, H, F/H]
        qkv_reshape = qkv.view(
            qkv.shape[0], mem_slots, self.n_heads, self.qkv_size)

        # [B, N, H, F/H] => [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)

        # [B, H, N, key_size], [B, H, N, key_size], [B, H, N, value_size]
        q, k, v = torch.split(
            qkv_transpose, [self.key_size, self.key_size, self.value_size], -1)

        # scale q with d_k, the dimensionality of the key vectors
        # TODO: sonnet use qkv_size, but isn't d_k the dim of key vector in the paper?
        # q *= (self.qkv_size ** -0.5)
        q = q * (self.key_size ** -0.5)

        # make it [B, H, N, N]
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)

        # output is [B, H, N, V]
        output = torch.matmul(weights, v)

        # [B, H, N, V] => [B, N, H, V] => [B, N, H*V]
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view(
            (output_transpose.shape[0], output_transpose.shape[1], -1))

        return new_memory


class NormMLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(NormMLP, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, activations):
        return self.layer_norm(self.linear(F.relu(activations)))


class ResidualTransform(nn.Module):

    def __init__(self, n_neighbours, key_size, value_size, n_heads, hidden_dim):
        super(ResidualTransform, self).__init__()
        self.attention = MHDPA(n_neighbours, key_size, value_size, n_heads)
        self.norm_mlp = NormMLP(hidden_dim, hidden_dim)

    def forward(self, activations):
        activations = activations + self.attention(activations)
        activations = activations + self.norm_mlp(activations)
        return activations


class RSAFFDecoder(nn.Module):
    """Relational Self-Attention Feed-Forward Decoder as defined in https://openreview.net/pdf?id=ByeSdsC9Km"""

    def __init__(self, n_classes, query_embed_dim, label_embed_dim, n_neighbours,
                 key_size, value_size, n_heads, num_layers):
        """
        Args:
            n_classes: the number of classes to classify
            query_embed_dim: the dimensionality of query embeddings
            label_embed_dim: the dimensionality of neighbour embeddings
            n_neighbours: the amount of neighbour queries fetched from memory store (= memory slots for RMC)
            key_size: the dimensionality of keys and queries in MHDPA
            value_size: the dimensionality of values in MHDPA
            n_heads: number of separate DPA blocks inside NHDPA
            num_layers: number of times Self-attention block is applied

        """
        super(RSAFFDecoder, self).__init__()
        self.n_classes = n_classes
        self.query_embed_dim = query_embed_dim
        self.n_neighbours = n_neighbours
        self.key_size = key_size
        self.value_size = value_size
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.label_embed_dim = label_embed_dim

        # Input is concatention of query embeddings, neighbout embeddings,
        # label embeddings and distances
        # (batch_size, n_neighbours, 3 * embed_dim + 1)
        self.input_size = self.label_embed_dim + 2 * self.query_embed_dim + 1
        self.hidden_dim = self.value_size * self.n_heads

        # Reserving one extra label for missing labels
        self.label_embeddings = nn.Embedding(
            self.n_classes + 1, self.label_embed_dim)
        self.pre_transform = nn.Linear(self.input_size, self.hidden_dim)
        residual_layers = [ResidualTransform(
            self.n_neighbours, self.key_size, self.value_size, self.n_heads,
            self.hidden_dim) for _ in range(self.num_layers)]
        self.residual_attention = nn.Sequential(*residual_layers)
        self.logits = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, buffer_embeds, buffer_labels, query, distances):
        """
        Args:
            buffer_embeds: torch tensor with (batch_size, n_neighbours,  embed_dim) size
            buffer_labels: torch tensor with (batch_size, n_neighbours) size
            query: torch tensor with (batch_size, embed_dim) size
        Returns:
            torch tensor with log probabilities - (batch_size, n_classes) size
        """
        # Calculating distances between queries and suggested neighbours
        query_copies = torch.cat(
            [query.unsqueeze(1)] * self.n_neighbours, dim=1)
        distances = F.softmax(-distances, dim=1)

        # Input preprocessing
        label_embeds = self.label_embeddings(buffer_labels)
        concat_inputs = torch.cat(
            [buffer_embeds, label_embeds, query_copies, distances.unsqueeze(-1)], dim=-1)

        # Unrolling MHDPA computations
        memory = self.pre_transform(concat_inputs)
        memory = self.residual_attention(memory)

        # Weighing memory by original distances
        # Normalizing distances to send "nice" inputs to softmax
        # Switching to -distance so that closest entries receive bigger weight
        weighted_memory = torch.sum(memory * distances.unsqueeze(-1), dim=1)

        logits = self.logits(weighted_memory)
        return logits


class BasicBlock(nn.Module):

    def __init__(self, stride, padding):
        super(BasicBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64, 64, 3, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(F.relu(self.batch_norm(x)))


class BlockCombo(nn.Module):

    def __init__(self):
        super(BlockCombo, self).__init__()
        self.block1 = BasicBlock(stride=2, padding=1)
        self.block2 = BasicBlock(stride=1, padding=1)
        self.block3 = BasicBlock(stride=1, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = out + self.block2(out)
        out = out + self.block3(out)
        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(1, 64, 3, padding=1)
        self.conv_seq = nn.Sequential(*[BlockCombo() for _ in range(5)])
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_seq(out)
        out = out.view(out.shape[0], -1)
        out = self.norm(out)
        return out
