import gin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from functools import partial
import logging

@gin.configurable('masking')
def parrallel_recomb(q_t, kv_t, att_type='all', local_context=3, bin_size=None):
    """ Return mask of attention matrix (ts_q, ts_kv) """
    with torch.no_grad():
        q_t[q_t == -1.0] = float('inf')  # We want padded to attend to everyone to avoid any nan.
        kv_t[kv_t == -1.0] = float('inf')  # We want no one to attend the padded values

        if bin_size is not None:  # General case where we use unaligned timesteps.
            q_t = q_t / bin_size
            starts_q = q_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            q_t -= starts_q
            kv_t = kv_t / bin_size
            starts_kv = kv_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            kv_t -= starts_kv

        bs, ts_q = q_t.size()
        _, ts_kv = kv_t.size()
        q_t_rep = q_t.view(bs, ts_q, 1).repeat(1, 1, ts_kv)
        kv_t_rep = kv_t.view(bs, 1, ts_kv).repeat(1, ts_q, 1)
        diff_mask = (q_t_rep - kv_t_rep).to(q_t_rep.device)
        if att_type == 'all':
            return (diff_mask >= 0).float()
        if att_type == 'local':
            return ((diff_mask >= 0) * (diff_mask <= local_context) + (diff_mask == float('inf'))).float()
        if att_type == 'strided':
            return ((diff_mask >= 0) * (torch.floor(diff_mask) % local_context == 0) + (
                    diff_mask == float('inf'))).float()


class PositionalEncoding(nn.Module):
    "Positiona Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"

    def __init__(self, emb, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2).float() * (-math.log(10000.0) / emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :]


class SelfAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.
    Input has shape (batch_size, n_timestemps, emb).

    ----------
    emb:
        Dimension of the input vector.
    hidden:
        Dimension of query, key, value matrixes.
    heads:
        Number of heads.

    mask:
        Mask the future timestemps
    """

    def __init__(self, emb, hidden, heads=8, mask=True, att_type='all', local_context=None, mask_aggregation='union',
                 dropout_att=0.0):
        """Initialize the Multi Head Block."""
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.hidden = hidden
        self.mask = mask
        self.drop_att = nn.Dropout(dropout_att)

        # Sparse transformer specific params
        self.att_type = att_type
        self.local_context = local_context
        self.mask_aggregation = mask_aggregation

        # Query, keys and value matrices
        self.w_keys = nn.Linear(emb, hidden * heads, bias=False)
        self.w_queries = nn.Linear(emb, hidden * heads, bias=False)
        self.w_values = nn.Linear(emb, hidden * heads, bias=False)

        # Output linear function
        self.unifyheads = nn.Linear(heads * hidden, emb)

    def forward(self, x):
        """
        x:
            Input data tensor with shape (batch_size, n_timestemps, emb)
        hidden:
            Hidden dim (dimension of query, key, value matrixes)

        Returns
            Self attention tensor with shape (batch_size, n_timestemps, emb)
        """
        # bs - batch_size, n - vectors number, emb - embedding dimensionality
        bs, n, emb = x.size()
        h = self.heads
        hidden = self.hidden

        keys = self.w_keys(x).view(bs, n, h, hidden)
        queries = self.w_queries(x).view(bs, n, h, hidden)
        values = self.w_values(x).view(bs, n, h, hidden)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        queries = queries.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        values = values.transpose(1, 2).contiguous().view(bs * h, n, hidden)

        # dive on the square oot of dimensionality
        queries = queries / (hidden ** (1 / 2))
        keys = keys / (hidden ** (1 / 2))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if self.mask:  # We deal with different masking and recombination types here
            if isinstance(self.att_type, list):  # Local and sparse attention
                if self.mask_aggregation == 'union':
                    mask_tensor = 0
                    for att_type in self.att_type:
                        mask_tensor += \
                        parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         att_type,
                                         self.local_context)[0]
                    mask_tensor = torch.clamp(mask_tensor, 0, 1)
                    dot = torch.where(mask_tensor.bool(),
                                      dot,
                                      torch.tensor(float('-inf')).cuda()).view(bs * h, n, n)

                elif self.mask_aggregation == 'split':

                    dot_list = list(torch.split(dot, dot.shape[0] // len(self.att_type), dim=0))
                    for i, att_type in enumerate(self.att_type):
                        mask_tensor = \
                        parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         att_type,
                                         self.local_context)[0]

                        dot_list[i] = torch.where(mask_tensor.bool(), dot_list[i],
                                                  torch.tensor(float('-inf')).cuda()).view(*dot_list[i].shape)
                    dot = torch.cat(dot_list, dim=0)
            else:  # Full causal masking
                mask_tensor = \
                parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                 torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                 self.att_type,
                                 self.local_context)[0]
                dot = torch.where(mask_tensor.bool(),
                                  dot,
                                  torch.tensor(float('-inf')).cuda()).view(bs * h, n, n)

        # dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(bs, h, n, hidden)

        # apply the dropout
        out = self.drop_att(out)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(bs, n, h * hidden)
        return self.unifyheads(out)


class SparseBlock(nn.Module):

    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True,
                 mask_aggregation='union', local_context=3, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, hidden, heads=heads, mask=mask, mask_aggregation=mask_aggregation,
                                       local_context=local_context, att_type=['strided', 'local'],
                                       dropout_att=dropout_att)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class LocalBlock(nn.Module):

    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True, local_context=3, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, hidden, heads=heads, mask=mask, mask_aggregation=None,
                                       local_context=local_context, att_type='local',
                                       dropout_att=dropout_att)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x
    
@gin.configurable("TransformerBlock")
class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        dropout=0.0,
        mask=True,
        dropout_att=0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            emb, hidden, heads=heads, mask=mask, dropout_att=dropout_att
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


@gin.configurable("StackedTransformerBlocks")
class StackedTransformerBlocks(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        depth: int = 1,
        dropout: float = 0.0,
        mask: bool = True,
        dropout_att=0.0,
    ):
        super().__init__()

        layers = [
            TransformerBlock(
                emb=emb,
                hidden=hidden,
                heads=heads,
                ff_hidden_mult=ff_hidden_mult,
                dropout=dropout,
                mask=mask,
                dropout_att=dropout_att,
            )
            for _ in range(depth)
        ]
        self.transformer_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.transformer_blocks(x)


# From TCN original paper https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation), dim=None)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation), dim=None)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class OrthogonalEmbeddings(nn.Module):
    """
    Creates an orthogonal basis for categorical embeddings
    A parameter free approach to encode categories; best
    paired with a bias term to represent the categorical itself.
    """

    def __init__(self, num_categories: int, token_dim: int) -> None:
        super().__init__()

        assert_msg = f"[{self.__class__.__name__}] require token dim {token_dim} >= num. cat. {num_categories}"
        assert token_dim >= num_categories, assert_msg

        random_mat = torch.randn(token_dim, token_dim)
        u_mat, _, vh_mat = torch.linalg.svd(random_mat)
        ortho_mat = u_mat @ vh_mat

        self.ortho_basis = ortho_mat[:num_categories, :]

    def forward(self, x: torch.Tensor):
        """
        Forward Method

        Parameter
        ---------
        x: torch.Tensor
            batch of one-hot encoded categorical values
            shape: [#batch, #classes]
        """
        return x @ self.ortho_basis.to(x.device)


@gin.configurable("FeatureTokenizer")
class FeatureTokenizer(nn.Module):

    """
    from Revisiting Deep Learning Models for Tabular Data

    x = (x_cat, x_num)
    x_num - (bs, sq, d_numerical)
    x_cat - (bs, sq, dim_cat), where dim_cat = sum(categories), already one-hot vectors!

    TEST:
    x_num = torch.rand(1,1,5)
    x_cat = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]])
    categories = [4, 6, 13, 5]
    d_token = 3

    ft = FeatureTokenizer(5,categories,3)
    res = ft((x_num, x_cat))

    assert res.size()== (1, 1, 10, 3) - result has shape (bs, sq, number of x_num features+number of x_cat features+CLS, d_token)
                                        number of x_num features = 5, number of x_cat features: 4 features with range from categories

    """

    def __init__(
        self,
        d_inp,
        d_token,
        categories,
        bias=True,
        categorical_embedding: str = "linear",
        use_cls=True,
    ):
        """
        Constructor for `FeatureTokenizer`

        Parameter
        ---------
        d_inp: int
            number of features to embed
        d_token: int
            embedding dimension
        categories: list[int]
            list of categorical variables and their cardinalities
        bias: bool
            whether to add and learn a bias term
        categorical_embedding: str
            form of embedding categoricals
            options: {linear, orthonormal}
        """
        super().__init__()

        self.categories = categories

        self.use_cls = use_cls

        # self.activation = torch.nn.Sigmoid()

        d_numerical = d_inp - sum(self.categories)

        if not self.categories:
            d_bias = d_numerical
        else:
            d_bias = d_numerical + len(categories)

        self.weight_num = nn.Parameter(
            torch.Tensor(d_numerical + self.use_cls, d_token)
        )  # +1 for CLS token
        nn.init.kaiming_uniform_(self.weight_num, a=math.sqrt(5))

        if categorical_embedding == "linear":
            categorical_module = partial(nn.Linear, bias=False)
        elif categorical_embedding == "orthonormal":
            logging.info(
                f"[{self.__class__.__name__}] using 'orthonormal' categorical embeddings"
            )
            categorical_module = OrthogonalEmbeddings
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] categorical embedding: {categorical_embedding} not impl."
            )

        self.weight_cat = [
            categorical_module(cat_i, d_token) for cat_i in self.categories
        ]
        self.weight_cat = nn.ModuleList(self.weight_cat)

        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num, x_cat):

        num_size = x_num.size(-1)
        cat_size = x_cat.size(-1)

        assert (num_size or cat_size) != 0

        if cat_size == 0:
            x_some = x_num
        else:
            x_some = x_cat

        bsXsq, _ = x_some.size()

        if self.use_cls:
            x_num = torch.cat(
                ([] if num_size == 0 else [x_num])
                + [torch.ones(bsXsq, 1, device=x_some.device)],
                dim=-1,
            )  # CLS token in the END!

        x = self.weight_num[None] * x_num[..., None]

        if self.categories:
            x_cat = torch.split(x_cat, self.categories, dim=-1)
            x_cat = [self.weight_cat[i](x_cat[i]) for i in range(len(x_cat))]
            x_cat = torch.stack(x_cat, -2)
            x = torch.cat([x_cat, x], -2)  # CLS token in the END!

        if self.bias is not None:
            bias = torch.cat(
                [
                    self.bias,
                    torch.zeros(1, self.bias.shape[1], device=x_some.device),
                ]
            )

            x = x + bias[None]
        return x


@gin.configurable("Bert_Head")
class Bert_Head(nn.Module):
    """BERT-like inference with CLS token."""

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = x[
            :, -1
        ]  # CLS token is the last one, second dim corresponds to the modalities
        x = self.normalization(x)
        x = self.relu(x)
        x = self.linear(x)
        return x


@gin.configurable("FeatureTokenizer_Transformer")
class FeatureTokenizer_Transformer(nn.Module):
    """FT-Transformer Embedding Module from Revisiting Deep Learning Models for Tabular Data."""

    def __init__(
        self,
        input_dim,
        token_dim,
        output_dim=None,
        categories=None,
        use_head=True,
        transformer=gin.REQUIRED,
    ):
        super().__init__()

        self.feature_tokens = FeatureTokenizer(
            input_dim, token_dim, categories=categories
        )
        self.transformer = transformer(token_dim)
        self.use_head = use_head
        if self.use_head:
            self.head = Bert_Head(token_dim, output_dim)

    def forward(self, x_num, x_cat):
        if len(x_num.size()) == 3:
            bs, sq, dim_num = x_num.size()
            bs, sq, dim_cat = x_cat.size()
            x_num = x_num.view(bs * sq, dim_num)
            x_cat = x_cat.view(bs * sq, dim_cat) # dim = features+1 for CLS token
        else:
            assert len(x_num.size()) == 2
        x = self.feature_tokens(x_num, x_cat)
        # x = self.transformer(x)  # (bs*sq, feature(+cls) token, feature_dim)
        # if len(x_num.size()) == 2:
        #     return x # (bs, feature(+cls) token, feature_dim)
        # x = self.head(x)  # (bs*sq, out_dim)
        # x = x.view(bs, sq, x.size(-1)) # (bs, sq, out_dim)

        return x