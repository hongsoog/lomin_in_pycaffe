import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.proj(x))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2) * \
                            -(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(1,0,2)

        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer_wrapper(nn.Module):
    def __init__(self, model, src_embed, tgt_embed, generator, max_len):
        super(Transformer_wrapper, self).__init__()
        self.model = model
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.ys_masks = list()
        for i in range(max_len+1):
            self.ys_masks.append(self.model.generate_square_subsequent_mask(i+1).cuda())
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model.forward(self.src_embed(src), self.tgt_embed(tgt),src_mask, tgt_mask)
    
    def greedy_decode(self, src, max_len):
        memory = self.model.encoder(self.src_embed(src))
        ys = torch.cuda.LongTensor(max_len + 2, src.size(1)).fill_(0)
        scores = torch.cuda.FloatTensor(src.size(1), max_len + 1)
        preds = torch.cuda.LongTensor(src.size(1), max_len + 1)
        for i in range(max_len + 1):
            out = self.model.decoder(self.tgt_embed(ys[:i + 1]), memory, self.ys_masks[i])
            prob = self.generator(out[-1])
            scores[:, i], preds[:, i] = torch.max(prob, dim=1)
            ys[i+1, :] = preds[:, i]
        return scores, preds

def build_transformer_native(num_classes,
                             vfea_len,
                             nhead=8,
                             num_encoder_layers=6,
                             num_decoder_layers=6,
                             dim_feedforward = 512,
                             dropout=0.1,
                             max_len=20,
                             no_recurrent=False):
    d_model = dim_feedforward
    max_len = max_len
    position0 = PositionalEncoding(d_model, dropout, vfea_len + 2) 
    position1 = PositionalEncoding(d_model, dropout, max_len + 2)
    if no_recurrent:
        tgt = nn.Sequential(nn.Embedding(max_len + 2, d_model), position1)
    else:
        tgt = nn.Sequential(nn.Embedding(num_classes, d_model), position1)
    src = position0
    transformer_model = nn.Transformer(d_model=dim_feedforward, 
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
    model = Transformer_wrapper(transformer_model, src, tgt, Generator(d_model, num_classes), max_len)

    return model

