from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pytorch_lightning import LightningModule
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report


# Define the Transformer Model with PyTorch Lightning
class TransformerModel(LightningModule):
    def __init__(self, num_heads, ff_dim, num_layers, input_dim, output_dim, dropout_rate=0.2, learning_rate=0.001):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access during training
        self.embedding_layer = nn.Linear(input_dim, input_dim)
        self.attention = MultiHeadSelfAttention(input_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_normal1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.layer_normal2 = nn.LayerNorm(input_dim, eps=1e-6)
        self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, output_dim)

        # Loss and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        attention_output, _ = self.attention(x)
        x = self.layer_normal1(x + self.dropout1(attention_output))
        x = self.layer_normal2(x + self.dropout2(self.ffn(x)))
        x = x.transpose(1, 2)  # Prepare for global average pooling
        x = self.global_average_pooling(x).squeeze(-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Define MultiHeadSelfAttention (used in the TransformerModel)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.projection_dim = embed_dim // num_heads

        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = key.size(-1)
        scaled_score = score / (dim_key ** 0.5)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention.view(batch_size, -1, self.embed_dim)
        output = self.combine_heads(concat_attention)
        return output, weights