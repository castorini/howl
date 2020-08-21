from typing import Any

from pydantic import BaseSettings
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RegisteredModel


__all__ = ['LASEncoderConfig',
           'FixedAttentionModuleConfig',
           'LASClassifierConfig',
           'LASEncoder',
           'LASClassifier']


class LASEncoderConfig(BaseSettings):
    num_mels: int = 40
    num_spec_channels: int = 3
    num_latent_channels: int = 8
    hidden_size: int = 96
    num_layers: int = 1
    use_maxpool: bool = True


class FixedAttentionModuleConfig(BaseSettings):
    num_heads: int = 4
    hidden_size: int = 96


class LASClassifierConfig(BaseSettings):
    dnn_size: int = 256
    dropout: float = 0.1
    las_config: LASEncoderConfig = LASEncoderConfig()
    fixed_attn_config: FixedAttentionModuleConfig = FixedAttentionModuleConfig()


class LstmConfig(BaseSettings):
    num_mels: int = 40
    hidden_size: int = 128
    num_labels: int = 2


class SequentialLstm(RegisteredModel, name='seq-lstm'):
    def __init__(self, num_labels: int, config: LstmConfig = LstmConfig()):
        super().__init__(num_labels)
        self.lstm = nn.LSTM(config.num_mels, config.hidden_size)
        self.dnn = nn.Sequential(nn.Linear(config.hidden_size, int(2 * config.hidden_size)),
                                 nn.ReLU(),
                                 nn.Linear(int(2 * config.hidden_size), num_labels))
        self.hc = None

    @property
    def streaming_state(self) -> Any:
        return self.hc

    @streaming_state.setter
    def streaming_state(self, x: Any):
        self.hc = x

    def forward(self, x, lengths):
        x = x[:, 0]  # Use log-Mels only
        hx = self.streaming_state if self.is_streaming and self.streaming_state is not None else None
        x = x.permute(2, 0, 1).contiguous()
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)
        rnn_seq, hc = self.lstm(x, hx=hx)
        if self.is_streaming:
            self.streaming_state = (hc[0].clone().detach(), hc[1].clone().detach())
        if lengths is not None:
            rnn_seq, _ = pad_packed_sequence(rnn_seq)
        return self.dnn(rnn_seq)


class SimpleLstm(RegisteredModel, name='lstm'):
    def __init__(self, num_labels: int, config: LstmConfig = LstmConfig()):
        super().__init__(num_labels)
        self.lstm = nn.LSTM(config.num_mels, config.hidden_size)
        self.dnn = nn.Sequential(nn.Linear(config.hidden_size, int(2 * config.hidden_size)),
                                 nn.ReLU(),
                                 nn.Linear(int(2 * config.hidden_size), num_labels))
        self.hc = None

    def forward(self, x, lengths):
        x = x[:, 0]  # Use log-Mels only
        hx = self.streaming_state if self.is_streaming and self.streaming_state is not None else None
        rnn_seq, hc = self.lstm(pack_padded_sequence(x.permute(2, 0, 1).contiguous(), lengths), hx=hx)
        if self.is_streaming:
            self.streaming_state = hc
        return self.dnn(hc[0].squeeze(0))


class SimpleGru(RegisteredModel, name='gru'):
    def __init__(self, num_labels: int, config: LASEncoderConfig = LASEncoderConfig()):
        super().__init__(num_labels)
        conv1 = nn.Conv2d(1, config.num_latent_channels, 3, padding=(1, 3))
        conv2 = nn.Conv2d(config.num_latent_channels, 1, 3, padding=1)
        self.conv_encoder = nn.Sequential(conv1,
                                          nn.BatchNorm2d(config.num_latent_channels),
                                          nn.ReLU(),
                                          nn.MaxPool2d((1, 2 if config.use_maxpool else 1)),
                                          conv2,
                                          nn.ReLU(),
                                          nn.BatchNorm2d(1))
        self.use_maxpool = config.use_maxpool
        self.lstm_encoder = nn.GRU(config.num_mels, config.hidden_size, bidirectional=False)
        self.dnn = nn.Sequential(nn.Linear(config.hidden_size, int(2 * config.hidden_size)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(int(2 * config.hidden_size), num_labels))

    def forward(self, x, lengths):
        if lengths is None:
            lengths = torch.tensor([x.size(-1)] * x.size(0)).to(x.device)
        x = x[:, :1]  # Use log-Mels only
        x = self.conv_encoder(x).squeeze(1)
        lengths += 4
        if self.use_maxpool:
            lengths = (lengths.float() / 2).floor()
        x = x.permute(2, 0, 1).contiguous()
        rnn_seq, rnn_out = self.lstm_encoder(pack_padded_sequence(x, lengths))
        if rnn_out.dim() > 2:
            bsz = rnn_out.size(1)
            rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(bsz, -1)
        return self.dnn(rnn_out)


class LASEncoder(nn.Module):
    def __init__(self, config: LASEncoderConfig):
        super().__init__()
        out_channels = config.num_latent_channels
        hidden_size = config.hidden_size
        self.use_maxpool = config.use_maxpool
        self.conv1 = conv1 = nn.Conv2d(config.num_spec_channels, stride=1, kernel_size=3, padding=2, out_channels=out_channels)
        self.conv2 = conv2 = nn.Conv2d(out_channels, stride=1, kernel_size=3, padding=2, out_channels=out_channels)
        self.conv_encoder = nn.Sequential(conv1,
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),
                                          nn.MaxPool2d((1, 2 if config.use_maxpool else 1)),
                                          conv2,
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(),
                                          nn.MaxPool2d((1, 2 if config.use_maxpool else 1)))
        self.lstm_encoder = nn.LSTM(out_channels * 44, hidden_size, config.num_layers, bias=True, bidirectional=True)

    def forward(self, x, lengths):
        if lengths is None:
            lengths = torch.tensor([x.size(-1)] * x.size(0)).to(x.device)
        x = self.conv_encoder(x)
        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        lengths = ((lengths.float() - self.conv1.kernel_size[1] + 4) / self.conv1.stride[1] + 1).floor()
        if self.use_maxpool:
            lengths = (lengths / 2).floor()
        lengths = ((lengths.float() - self.conv2.kernel_size[1] + 4) / self.conv2.stride[1] + 1).floor()
        if self.use_maxpool:
            lengths = (lengths / 2).floor()
        rnn_seq, (rnn_out, _) = self.lstm_encoder(pack_padded_sequence(x, lengths))
        return rnn_seq, rnn_out


class FixedAttentionModule(nn.Module):
    def __init__(self, config: FixedAttentionModuleConfig):
        super().__init__()
        self.context_vec = nn.Parameter(torch.Tensor(config.hidden_size * 2).uniform_(-0.25, 0.25), requires_grad=True)
        self.v_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.k_proj = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.num_heads = config.num_heads

    def forward(self, rnn_seq, mask=None):
        values = self.v_proj(rnn_seq)
        keys = self.k_proj(rnn_seq)
        rnn_seq = values.view(values.size(0), values.size(1), self.num_heads, values.size(2) // self.num_heads)
        keys = keys.view(values.size(0), keys.size(1), self.num_heads, keys.size(2) // self.num_heads)
        cvec = self.context_vec.view(-1, self.num_heads).unsqueeze(-1).expand(-1, -1, rnn_seq.size(0))
        logits = torch.einsum('ijkl,lki->ijk', rnn_seq, cvec)
        if mask is not None:
            mask = (1 - mask) * -100
            logits += mask.unsqueeze(-1).expand_as(logits)
        scores = F.softmax(logits, 0)
        vec = torch.einsum('ijk,ijkl->jkl', scores, keys)
        return vec.view(vec.size(0), -1)


class LASClassifier(RegisteredModel, name='las'):
    def __init__(self, num_labels: int, config: LASClassifierConfig = LASClassifierConfig()):
        super().__init__(num_labels)
        self.encoder = LASEncoder(config.las_config)
        self.attn = FixedAttentionModule(config.fixed_attn_config)
        self.fc = nn.Sequential(nn.Linear(config.las_config.hidden_size * 2, config.dnn_size),
                                nn.ReLU(),
                                nn.Dropout(config.dropout),
                                nn.Linear(config.dnn_size, num_labels))

    def forward(self, x, lengths):
        rnn_seq, rnn_out = self.encoder(x, lengths)
        rnn_seq, lengths = pad_packed_sequence(rnn_seq)

        mask = torch.zeros(rnn_seq.size(0), rnn_seq.size(1))
        for idx, length in enumerate(lengths.tolist()):
            mask[:length, idx] = 1
        mask = mask.to(rnn_seq.device)
        context = self.attn(rnn_seq, mask=mask)
        return self.fc(context)
