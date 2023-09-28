import gin
import numpy as np
import torch
import torch.nn as nn
from rtdl import FeatureTokenizer
from icu_benchmarks.models.layers import TransformerBlock, LocalBlock, parrallel_recomb,\
    TemporalBlock, SparseBlock, PositionalEncoding


@gin.configurable('LSTM')
class LSTMNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, h = self.rnn(x, (h0, c0))
        pred = self.logit(out)
        return pred


@gin.configurable('GRU')
class GRUNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)

        return pred


@gin.configurable('Transformer')
class Transformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 pos_encoding=True, dropout_att=0.0, proj=False, ckpt_path=None, cluster=False):
        super().__init__()

        self.proj = proj
        self.cluster = cluster
        if proj:
            self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                            ff_hidden_mult=ff_hidden_mult,
                                            dropout=dropout, dropout_att=dropout_att))

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['model'])

    def forward(self, x):
        if self.proj:
            x = self.input_embedding(x)
            if self.cluster:
                return x
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred


@gin.configurable('LocalTransformer')
class LocalTransformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 pos_encoding=True, local_context=1, dropout_att=0.0):
        super().__init__()

        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(LocalBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                      ff_hidden_mult=ff_hidden_mult, local_context=local_context,
                                      dropout=dropout, dropout_att=dropout_att))

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred


@gin.configurable('NaiveSparseTransformer')
class NaiveSparseTransformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 mask_aggregation='union', local_context=3, pos_encoding=True, dropout_att=0.0):
        super().__init__()
        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults

        tblocks = []
        for i in range(depth):
            tblocks.append(SparseBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                       ff_hidden_mult=ff_hidden_mult, dropout=dropout,
                                       mask_aggregation=mask_aggregation, local_context=local_context,
                                       dropout_att=dropout_att))
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)
        w_input = list(self.input_embedding.parameters())[0]
        l1_norm_input = torch.torch.norm(w_input, 1)

        return pred, l1_norm_input * self.l1_reg


# From TCN original paper https://github.com/locuslab/TCN
@gin.configurable('TCN')
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes,
                 max_seq_length=0, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, int) and max_seq_length:
            num_channels = [num_channels] * int(np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size)))
        elif isinstance(num_channels, int) and not max_seq_length:
            raise Exception('a maximum sequence length needs to be provided if num_channels is int')

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to channel first
        o = self.network(x)
        o = o.permute(0, 2, 1)  # Permute to channel last
        pred = self.logit(o)
        return pred

# RTDL model for continuous bag of words pretraining task
@gin.configurable('RTDL')
class RTDL(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, emb_dim):
        '''
        n_num_features: int. the number of continuous features
        cat_cardinalities: list of ints. the cardinality of each categorical feature
        emb_dim: int. the embedding dimension
        '''
        super(RTDL, self).__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.max_cat_cardinality = max(cat_cardinalities)
        self.enc = FeatureTokenizer(n_num_features, cat_cardinalities, emb_dim)
        self.dec_num = nn.ModuleList([nn.Linear(emb_dim, 1) for _ in range(n_num_features)])
        self.dec_cat = nn.ModuleList([nn.Linear(emb_dim, cat_cardinality) for cat_cardinality in cat_cardinalities])

    def forward(self, x_num, x_cat, mask_num, mask_cat, pred_idx_num, pred_idx_cat):
        '''
        x_num: tensor of shape (batch_size, n_num_features)
        x_cat: tensor of shape (batch_size, n_cat_features)
        mask_num: tensor of shape (batch_size, n_features), 0 for missing values
        mask_cat: tensor of shape (batch_size, n_features), 0 for missing values
        pred_idx_num: tensor of shape (batch_size, 1), the index of the predicted continuous feature
        pred_idx_cat: tensor of shape (batch_size, 1), the index of the predicted categorical feature
        '''
        if x_num.shape[1] == self.n_num_features:
            embeds = self.enc(x_num, x_cat) # (batch_size, n_features, emb_dim)
        else: # consider previous timestep
            assert x_num.shape[1] == self.n_num_features * 2
            embeds_1 = self.enc(x_num[:, :self.n_num_features], x_cat[:, :len(self.cat_cardinalities)]) # (batch_size, n_features, emb_dim)
            embeds_2 = self.enc(x_num[:, self.n_num_features:], x_cat[:, len(self.cat_cardinalities):]) # (batch_size, n_features, emb_dim)
            embeds = torch.cat([embeds_1, embeds_2], dim=1) # (batch_size, 2 * n_features, emb_dim)
        if mask_num is None and mask_cat is None:  # for fine-tuning  
            return embeds
        pred_num = None
        pred_cat = None
        if mask_num is not None:
            assert pred_idx_num.shape[0] == x_num.shape[0]
            mask_num = mask_num.unsqueeze(-1) # (batch_size, n_features, 1)
            embeds_num = embeds * mask_num  # (batch_size, n_features, emb_dim)
            embeds_num = torch.sum(embeds_num, dim=1) / torch.sum(mask_num, dim=1) # (batch_size, emb_dim)
            pred_num = torch.cat([self.dec_num[pred_idx_num[i]](embeds_num[i:i+1]) for i in range(len(pred_idx_num))], dim=0) # (batch_size, 1)
        if mask_cat is not None:
            assert pred_idx_cat.shape[0] == x_cat.shape[0]
            mask_cat = mask_cat.unsqueeze(-1) # (batch_size, n_features, 1)
            embeds_cat = embeds * mask_cat  # (batch_size, n_features, emb_dim)
            embeds_cat = torch.sum(embeds_cat, dim=1) / torch.sum(mask_cat, dim=1) # (batch_size, emb_dim)
            pred_cat = [self.dec_cat[pred_idx_cat[i]-self.n_num_features](embeds_cat[i:i+1]) for i in range(len(pred_idx_cat))]
            paddings = [torch.ones((1, self.max_cat_cardinality - pred_cat[i].shape[1])) * (-100.0) for i in range(len(pred_idx_cat))]
            pred_cat_pad = [torch.cat([pred_cat[i], paddings[i].cuda()], dim=1) for i in range(len(pred_idx_cat))] 
            pred_cat = torch.cat(pred_cat_pad, dim=0) # (batch_size, max_cat_cardinality)
        return pred_num, pred_cat

# TransformerMLM for masked language modeling pretraining task
@gin.configurable('TransformerMLM')
class TransformerMLM(nn.Module):
    def __init__(self, encoder, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 pos_encoding=True, dropout_att=0.0, proj=False, ckpt_path=None, cluster=False, add_cls=False, cls_only=False):
        super().__init__()
        self.add_cls = add_cls
        self.cls_only = cls_only
        if add_cls:
            self.cls_token_embedding = nn.Parameter(torch.empty(1, 1, emb))
            nn.init.kaiming_uniform_(self.cls_token_embedding)
        self.mask_token_embedding = nn.Parameter(torch.empty(1, 1, emb))
        nn.init.kaiming_uniform_(self.mask_token_embedding)
        self.rtdl = encoder
        self.proj = proj
        self.cluster = cluster
        if proj:
            self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=hidden, hidden=hidden, heads=heads, mask=False,
                                            ff_hidden_mult=ff_hidden_mult,
                                            dropout=dropout, dropout_att=dropout_att))

        self.tblocks = nn.Sequential(*tblocks)
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['model'])

    def forward(self, x_num, x_cat, mask_num, mask_cat, pred_idx_num, pred_idx_cat):
        '''
        x_num: tensor of shape (batch_size, n_num_features)
        x_cat: tensor of shape (batch_size, n_cat_features)
        mask_num: tensor of shape (batch_size, n_features), 0 for missing values
        mask_cat: tensor of shape (batch_size, n_features), 0 for missing values
        pred_idx_num: tensor of shape (batch_size, 1), the index of the predicted continuous feature
        pred_idx_cat: tensor of shape (batch_size, 1), the index of the predicted categorical feature
        '''
        x = self.rtdl.enc(x_num, x_cat)
        if mask_num is None and mask_cat is None:  # for fine-tuning  
            return x
        batch_size, n_features, emb_dim = x.shape
        # 80% of the time, we replace the position of pred_idx_num and pred_idx_cat with the mask token ([MASK])
        # 10% of the time, we replace the position of pred_idx_num and pred_idx_cat with a random token
        # 10% of the time, we keep the position of pred_idx_num and pred_idx_cat unchanged
        if mask_num is not None and mask_cat is not None:
            mask_token_embedding = self.mask_token_embedding.expand(batch_size, n_features, emb_dim)  # now of shape (batch_size, n_features, emb_dim)
            mask_num = mask_num.unsqueeze(-1).expand_as(x) # (batch_size, n_features, emb_dim)
            mask_cat = mask_cat.unsqueeze(-1).expand_as(x) # (batch_size, n_features, emb_dim)
            random_prob = torch.rand(batch_size, 1).repeat((1, n_features)).unsqueeze(-1).cuda() # (batch_size, n_features, 1)
            mask_1 = random_prob < 0.8 # (batch_size, 1)
            x[~mask_num & mask_1] = mask_token_embedding[~mask_num & mask_1]
            x[~mask_cat & mask_1] = mask_token_embedding[~mask_cat & mask_1]
            mask_2 = (random_prob >= 0.8) & (random_prob < 0.9)
            random_token_num = torch.randn_like(x).cuda()
            random_token_cat = torch.randn_like(x).cuda()
            x[~mask_num & mask_2] = random_token_num[~mask_num & mask_2]
            x[~mask_cat & mask_2] = random_token_cat[~mask_cat & mask_2]
        
        if self.add_cls:
            cls_token_embedding = self.cls_token_embedding.expand(batch_size, 1, emb_dim)
            x = torch.cat([cls_token_embedding, x], dim=1) # (batch_size, n_features+1, emb_dim)

        if self.proj:
            x = self.input_embedding(x)
        if self.cluster:
            return x
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        x = self.tblocks(x) # (batch_size, n_features(+1), emb_dim)

        if self.add_cls and not self.cls_only:
            idx_num_used = pred_idx_num + 1
            idx_cat_used = pred_idx_cat - self.rtdl.n_num_features + 1
        if self.add_cls and self.cls_only:
            idx_num_used = torch.zeros_like(pred_idx_num, dtype=torch.long)
            idx_cat_used = torch.zeros_like(pred_idx_cat, dtype=torch.long)
        if not self.add_cls:
            idx_num_used = pred_idx_num
            idx_cat_used = pred_idx_cat - self.rtdl.n_num_features

        pred_num = torch.cat([self.rtdl.dec_num[pred_idx_num[i]](x[i:i+1, idx_num_used[i], :].squeeze(1)) for i in range(batch_size)], dim=0) # (batch_size, 1)
        pred_cat = [self.rtdl.dec_cat[pred_idx_cat[i]-self.rtdl.n_num_features](x[i:i+1, idx_cat_used[i], :].squeeze(1)) for i in range(batch_size)] 
        paddings = [torch.ones((1, self.rtdl.max_cat_cardinality - pred_cat[i].shape[1])) * (-100.0) for i in range(batch_size)]
        pred_cat_pad = [torch.cat([pred_cat[i], paddings[i].cuda()], dim=1) for i in range(batch_size)]
        pred_cat = torch.cat(pred_cat_pad, dim=0) # (batch_size, max_cat_cardinality)

        return pred_num, pred_cat
    
@gin.configurable('RTDLTransformer')
class RTDLTransformer(nn.Module):
    def __init__(self, encoder, decoder, pooling, ckpt_path=None, freeze=False):
        super(RTDLTransformer, self).__init__()
        self.encoder = encoder
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.encoder.load_state_dict(torch.load(ckpt_path)['model'])
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        self.decoder = decoder
        self.pooling = pooling
        
    def forward(self, x_num, x_cat, impute_mask=None):
        '''
        x_num: tensor of shape (batch_size, seq_len, n_num_features)
        x_cat: tensor of shape (batch_size, seq_len, n_cat_features)
        impute_mask: tensor of shape (batch_size, seq_len, ), 0 for missing values
        '''
        bsz, seq_len, n_num_features = x_num.shape
        _, _, n_cat_features = x_cat.shape
        n_features = n_num_features + n_cat_features
        embeds = self.encoder.enc(x_num.reshape(-1, n_num_features), x_cat.reshape(-1, n_cat_features)) # (batch_size * seq_len, n_features, emb_dim)

        if impute_mask is not None:
            embeds = embeds * impute_mask.reshape(-1, n_features, 1)
        embeds = embeds.reshape(bsz, seq_len, n_features, -1) # (batch_size, seq_len, n_features, emb_dim)
        if self.pooling == 'mean':
            embeds = torch.mean(embeds, dim=2) # (batch_size, seq_len, emb_dim)
        elif self.pooling == 'sum':
            embeds = torch.sum(embeds, dim=2) # (batch_size, seq_len, emb_dim)
        elif self.pooling == 'max':
            embeds = torch.max(embeds, dim=2)[0] # (batch_size, seq_len, emb_dim)
        elif self.pooling == "concat":
            embeds = embeds.reshape(bsz, seq_len, -1) # (batch_size, seq_len, emb_dim * n_features)
        else:
            raise NotImplementedError
        
        if self.decoder == "cluster":
            return embeds

        pred = self.decoder(embeds) # (batch_size, seq_len, 1)
        return pred
    

@gin.configurable('TransformerMLMTransformer')
class TransformerMLMTransformer(nn.Module):
    def __init__(self, encoder, decoder, pooling, ckpt_path=None, freeze=False, use_cls=False):
        super(TransformerMLMTransformer, self).__init__()
        self.encoder = encoder
        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.encoder.load_state_dict(torch.load(ckpt_path)['model'])
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        self.decoder = decoder
        self.pooling = pooling
        self.use_cls = use_cls
        
    def forward(self, x_num, x_cat, impute_mask=None):
        '''
        x_num: tensor of shape (batch_size, seq_len, n_num_features)
        x_cat: tensor of shape (batch_size, seq_len, n_cat_features)
        impute_mask: tensor of shape (batch_size, seq_len, ), 0 for missing values
        '''
        bsz, seq_len, n_num_features = x_num.shape
        _, _, n_cat_features = x_cat.shape
        n_features = n_num_features + n_cat_features
        embeds = self.encoder(x_num.reshape(-1, n_num_features), x_cat.reshape(-1, n_cat_features), None, None, None, None) # (batch_size * seq_len, n_features, emb_dim)
        if self.use_cls:
            embeds = embeds[:, 0, :].unsqueeze(1).reshape(bsz, seq_len, -1) # (batch_size, seq_len, emb_dim)
        else:
            if impute_mask is not None:
                embeds = embeds * impute_mask.reshape(-1, n_features, 1)
            embeds = embeds.reshape(bsz, seq_len, n_features, -1) # (batch_size, seq_len, n_features, emb_dim)
            if self.pooling == 'mean':
                embeds = torch.mean(embeds, dim=2) # (batch_size, seq_len, emb_dim)
            elif self.pooling == 'sum':
                embeds = torch.sum(embeds, dim=2) # (batch_size, seq_len, emb_dim)
            elif self.pooling == 'max':
                embeds = torch.max(embeds, dim=2)[0] # (batch_size, seq_len, emb_dim)
            elif self.pooling == "concat":
                embeds = embeds.reshape(bsz, seq_len, -1) # (batch_size, seq_len, emb_dim * n_features)
            else:
                raise NotImplementedError
        
        if self.decoder == "cluster":
            return embeds

        pred = self.decoder(embeds) # (batch_size, seq_len, 1)
        return pred







        
