from collections import OrderedDict
import torch
from torch import nn
from catalyst.contrib.models import ResnetEncoder
from catalyst.contrib.models import SequentialNet
from catalyst.dl.initialization import create_optimal_inner_init, outer_init
from catalyst.dl import registry


class TSN(nn.Module):
    def __init__(
            self,
            encoder,
            n_cls,
            feature_net_hiddens=None,
            emb_net_hiddens=None,
            activation_fn=torch.nn.ReLU,
            norm_fn=None,
            bias=True,
            dropout=None,
            consensus=None,
            kernel_size=1,
            feature_net_skip_connection=False,
            early_consensus=True):
        super().__init__()

        assert consensus is not None
        assert kernel_size in [1, 3, 5]

        consensus = consensus if isinstance(consensus, list) else [consensus]
        self.consensus = consensus

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.feature_net_skip_connection = feature_net_skip_connection
        self.early_consensus = early_consensus

        nonlinearity = registry.MODULES.get_if_str(activation_fn)
        inner_init = create_optimal_inner_init(nonlinearity=nonlinearity)
        kernel2pad = {1: 0, 3: 1, 5: 2}

        def layer_fn(in_features, out_features, bias=True):

            return nn.Conv1d(
                in_features, out_features, bias=bias,
                kernel_size=kernel_size, padding=kernel2pad[kernel_size])

        if feature_net_hiddens is not None:
            self.feature_net = SequentialNet(
                hiddens=[encoder.out_features] + [feature_net_hiddens],
                activation_fn=activation_fn,
                layer_fn=layer_fn,
                norm_fn=norm_fn, bias=bias, dropout=dropout)
            self.feature_net.apply(inner_init)
            out_features = feature_net_hiddens
        else:
            # if no feature net, then no need of skip connection (nothing to skip)
            assert not self.feature_net_skip_connection
            self.feature_net = lambda x: x
            out_features = encoder.out_features

        # Differences are starting here

        # Input channels to consensus function (also to embedding net multiplied by len(consensus))
        if self.feature_net_skip_connection:
            in_channels = out_features + encoder.out_features
        else:
            in_channels = out_features

        consensus_fn = OrderedDict()
        for key in sorted(consensus):
            if key == "attention":
                self.attn = nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=1,
                        kernel_size=kernel_size,
                        padding=kernel2pad[kernel_size],
                        bias=True),
                    nn.Softmax(dim=1))

                def self_attn_fn(x):
                    x_a = x.transpose(1, 2)
                    x_attn = (self.attn(x_a) * x_a)
                    x_attn = x_attn.transpose(1, 2)
                    x_attn = x_attn.mean(1, keepdim=True)
                    return x_attn

                consensus_fn["attention"] = self_attn_fn
            elif key == "avg":
                consensus_fn[key] = lambda x: x.mean(1, keepdim=True)
            elif key == "max":
                consensus_fn[key] = lambda x: x.max(1, keepdim=True)[0]

        # Not optimized if too more understandable logic
        if self.early_consensus:
            out_features = emb_net_hiddens

            self.emb_net = SequentialNet(
                hiddens=[in_channels * len(consensus_fn), emb_net_hiddens],
                activation_fn=activation_fn,
                norm_fn=norm_fn, bias=bias, dropout=dropout)
            self.emb_net.apply(inner_init)
        else:


            if self.feature_net_skip_connection:
                out_features = out_features + self.encoder.out_features
            else:
                out_features = out_features

        self.head = nn.Linear(out_features, n_cls, bias=True)

        if 'attention' in consensus:
            self.attn.apply(outer_init)
        self.head.apply(outer_init)

        self.consensus_fn = consensus_fn


    def forward(self, input):
        if len(input.shape) < 5:
            input = input.unsqueeze(1)
        bs, fl, ch, h, w = input.shape
        x = input.view(-1, ch, h, w)
        x = self.encoder(x)
        x = self.dropout(x)
        identity = x

        # in simple case feature_net is identity mapping
        x = x.view(bs, fl, -1)
        x = x.transpose(1, 2)
        x = self.feature_net(x)
        x = x.transpose(1, 2).contiguous()  # because conv1d
        x = x.view(bs * fl, -1)
        if self.feature_net_skip_connection:
            x = torch.cat([identity, x], dim=-1)
        else:
            x = x

        if self.early_consensus:
            x = x.view(bs, fl, -1)
            c_list = []

            for c_fn in self.consensus_fn.values():
                c_res = c_fn(x)
                c_list.append(c_res)
            x = torch.cat(c_list, dim=1)
            x = x.view(bs, -1)
            x = self.emb_net(x)

        x = self.head(x)

        if not self.early_consensus:
            x = x.view(bs, fl, -1)

            if self.consensus[0] == "avg":
                x = x.mean(1, keepdim=False)
            elif self.consensus[0] == "attention":
                identity = identity.view(bs, fl, -1)
                x_a = identity.transpose(1, 2)
                x_ = x.transpose(1, 2)
                x_attn = (self.attn(x_a) * x_)
                x_attn = x_attn.transpose(1, 2)
                x = x_attn.sum(1, keepdim=False)

        x = torch.sigmoid(x)  # with bce loss
        return x


def prepare_tsn_base_model(partial_bn=None, **kwargs):
    """
    :param partial_bn: 2 if partial_bn else 1
    :param kwargs:
    :return:
    """
    base_model = ResnetEncoder(**kwargs)
    if partial_bn is not None:
        count = 0
        for m in base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= partial_bn:
                    m.eval()

                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    return base_model


def tsn(base_model, tsn_model):
    base_model = prepare_tsn_base_model(**base_model)
    net = TSN(encoder=base_model, **tsn_model)
    return net
