"""
Stylization Module
              Encoder
Input Motion --------- Latent Embed ----
                             |
                             |-------
"""

# High level
# a style encoder that encode the style 
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

BASEPATH = os.path.dirname(__file__)
from os.path import join as pjoin


def get_conv_pad(kernel_size, stride, padding=nn.ReflectionPad1d):
    pad_l = (kernel_size - stride) // 2
    pad_r = (kernel_size - stride) - pad_l
    return padding((pad_l, pad_r))


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode = self.mode)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def ZeroPad1d(sizes):
    return nn.ConstantPad1d(sizes, 0)


def ConvLayers(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', use_bias=True):

    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    if pad_type == 'reflect':
        pad = nn.ReflectionPad1d
    elif pad_type == 'replicate':
        pad = nn.ReplicationPad1d
    elif pad_type == 'zero':
        pad = ZeroPad1d
    else:
        assert 0, "Unsupported padding type: {}".format(pad_type)

    pad_l = (kernel_size - 1) // 2
    pad_r = kernel_size - 1 - pad_l
    return [pad((pad_l, pad_r)),
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride, bias=use_bias)]


def get_acti_layer(acti='relu', inplace=True):

    if acti == 'relu':
        return [nn.ReLU(inplace=inplace)]
    elif acti == 'lrelu':
        return [nn.LeakyReLU(0.2, inplace=inplace)]
    elif acti == 'tanh':
        return [nn.Tanh()]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        # return [nn.InstanceNorm1d(norm_dim, affine=False)]  # for rt42!
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []


def ConvBlock(kernel_size, in_channels, out_channels, stride=1, pad_type='reflect', dropout=None,
              norm='none', acti='lrelu', acti_first=False, use_bias=True, inplace=True):
    """
    returns a list of [pad, conv, norm, acti] or [acti, pad, conv, norm]
    """

    layers = ConvLayers(kernel_size, in_channels, out_channels, stride=stride, pad_type=pad_type, use_bias=use_bias)
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_channels)
    acti_layers = get_acti_layer(acti, inplace=inplace)

    if acti_first:
        return acti_layers + layers
    else:
        return layers + acti_layers


def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


class ResBlock(nn.Module):
    def __init__(self, kernel_size, channels, stride=1, pad_type='zero', norm='none', acti='relu'):
        super(ResBlock, self).__init__()
        layers = []
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti=acti)
        layers += ConvBlock(kernel_size, channels, channels,
                            stride=stride, pad_type=pad_type,
                            norm=norm, acti='none')

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[: , : m.num_features]
            std = adain_params[: , m.num_features: 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[: , 2 * m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

class Config:
    # encoder style parameters
    channels = [0, 96, 144]
    kernel_size = 8
    stride = 1

    # mlp parameters
    mlp_dims = [144, 192, 256]


class StyleEncoder(nn.Module):
    def __init__(self, input_feats, latent_dim, nhead, ff_size, 
                 dropout, activation, num_layers, num_styles, enc='transformer'):
        super().__init__()
        # self.global_pool = F.max_pool1d

        # layers = []
        # n_convs = len(channels) - 1

        # for i in range(n_convs):
        #     layers += ConvBlock(kernel_size, channels[i], channels[i + 1],
        #                         stride=stride, norm='none', acti='lrelu')

        # self.conv_model = nn.Sequential(*layers)
        # self.channels = channels

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.num_styles = num_styles

        self.embed = nn.Sequential([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        ])

        if enc == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                      nhead=self.nhead,
                                                      dim_feedforward=self.ff_size,
                                                      dropout=self.dropout,
                                                      activation=self.activation)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.output = nn.Linear(self.latent_dim, self.num_styles)
        

    def forward(self, x):
        # x = self.conv_model(x)
        # kernel_size = x.shape[-1]
        # x = self.global_pool(x, kernel_size)
        # return x
        bs, timesteps, num_feats = x.shape
        x = x.permute(1, 0, 2)

        emb = self.embed(x)
        cls_prompt = torch.zeros((bs, 1, num_feats))
        emb = torch.cat([x, cls_prompt], dim=0)
        cls_latent = self.encoder(emb)[-1]
        cls = self.output(cls_latent)
        return cls_latent, cls
        
    

class ContentEncoder(nn.Module):
    def __init__(self, input_feats, latent_dim, nhead, ff_size, 
                 dropout, activation, num_layers, num_contents, enc='transformer'):
        super().__init__()
        # channels = config.enc_co_channels
        # kernel_size = config.enc_co_kernel_size
        # stride = config.enc_co_stride

        # layers = []
        # n_convs = config.enc_co_down_n
        # n_resblk = config.enc_co_resblks
        # acti = 'lrelu'

        # assert n_convs + 1 == len(channels)

        # for i in range(n_convs):
        #     layers += ConvBlock(kernel_size, channels[i], channels[i + 1],
        #                         stride=stride, norm='in', acti=acti)

        # for i in range(n_resblk):
        #     layers.append(ResBlock(kernel_size, channels[-1], stride=1,
        #                            pad_type='reflect', norm='in', acti=acti))

        # self.conv_model = nn.Sequential(*layers)
        # self.channels = channels

        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.num_contents = num_contents

        self.embed = nn.Sequential([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        ])

        if enc == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                      nhead=self.nhead,
                                                      dim_feedforward=self.ff_size,
                                                      dropout=self.dropout,
                                                      activation=self.activation)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.norm = nn.InstanceNorm1d(self.latent_dim)
        self.output = nn.Linear(self.latent_dim, self.num_styles)


    def forward(self, x):
        # x = self.conv_model(x)
        # return x
        emb = self.embed(x)
        latent = self.encoder(emb)
        latent = self.norm(latent)
        cls = self.output(latent)
        return latent, cls


class MLP(nn.Module):
    def __init__(self, dims, out_dim):
        super(MLP, self).__init__()
        n_blk = len(dims)
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim,
                                   norm='none', acti='none')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
    
class StyleEmbedding(nn.Module):
    def __init__(self, channels, kernel_size, stride, dims, out_dim):
        super().__init__()
        self.encoder = StyleEncoder(channels, kernel_size, stride)
        self.mlp = MLP(dims, out_dim)

    def forward(self, style):
        return self.mlp(self.encoder(style))

    
def recursive_print(data, prefix='  '):
    for key, value in data.items():
        if isinstance(value, dict):
            recursive_print(data, prefix + prefix)
        print(f"{prefix}key: {key} value: {value}")

def load_by_pattern(parameters, pattern: str=''):
    for layer_key, value in parameters.items():
        print(layer_key)

from models import TimestepEmbedder, PositionalEncoding, OutputProcess, TrajectorEncoder, MotionEncoder


class MotionLatentEncoder(nn.Module):
    def __init__(self, input_feats, trans_input, pose_input, latent_dim):
        super().__init__()
        # trajectory encoder
        self.traj_encoder = TrajectorEncoder(trans_input, pose_input, latent_dim)

        # motion encoder
        self.motion_encoder = MotionEncoder(input_feats, latent_dim)

    def forward(self, x, traj_trans, traj_pose, past_motion):
        traj_latent = self.traj_encoder(traj_trans, traj_pose)
        motion_latent = self.motion_encoder(x, past_motion)

        output = torch.cat([traj_latent, motion_latent], dim=0)
        return output


class MotionLatentDiffusion(nn.Module):
    def __init__(self, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 activation="gelu", arch='trans_enc', device=None):
        super().__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.ff_size = ff_size
        self.activation = activation
        self.arch = arch

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')


    def forward(self, timesteps, embed, nframes):
        time_emb = self.embed_timestep(timesteps)

        xseq = torch.cat((time_emb, embed), dim=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        return output
    

class MotionLatentDecoder(nn.Module):
    def __init__(self, latent_dim, output_feats, njoints, nfeats):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_feats = output_feats
        self.njoints = njoints
        self.nfeats = nfeats

        self.decoder = OutputProcess(self.output_feats, self.latent_dim, self.njoints, self.nfeats)

    def forward(self, latent_embed):
        output = self.decoder(latent_embed)
        return output
    

class Stylization(nn.Module):
    def __init__(self, stylizer='adain'):
        super().__init__()

        self.stylizer_type = stylizer
        self.style_encoder = StyleEncoder()
        self.content_encoder = ContentEncoder()

        self.encoder = MotionLatentEncoder()
        self.frozen_diff = MotionLatentDiffusion()
        self.decoder = MotionLatentDecoder()

        self.diff = MotionLatentDiffusion()

        if self.stylizer_type == 'adain':
            self.stylizer = AdaptiveInstanceNorm1d()
        elif self.stylizer_type is None:
            self.stylizer = None
        else:
            raise ValueError(f'Unsupported stylizer type {stylizer}')
        
    def load_pretrained(self, pth):
        raise NotImplementedError
    
    def extract_style_code(self, style_motion):
        return self.style_encoder(style_motion)
        
    def extract_content_code(self, content_motion):
        return self.content_encoder(content_motion)
    
    def init_stylizer(self, initalizer):
        if self.stylizer_type == 'adain':
            assign_adain_params(initalizer, self.stylizer)
        
    def forward(self, timestep, x, traj_trans, traj_pose, past_motion, style_code):
        bs, njoints, nfeats, nframes = x.shape

        self.init_stylizer(style_code)

        latent_feats = self.encoder(x, traj_trans, traj_pose, past_motion)
        unstylized = self.frozen_diff(timestep, latent_feats, nframes)
        stylized = self.diff(timestep, latent_feats, nframes)
        stylized = self.stylizer(stylized)

        output = self.decoder(unstylized + stylized)
        return output

if __name__ == '__main__':
    parameters = torch.load(pjoin(BASEPATH, '../checkpoints/stylization/gen_00400000.pt'))
    # recursive_print(parameters)
    # load_by_pattern(parameters)
    cfg = Config()
    model = StyleEmbedding(cfg, 256)
    print(model.state_dict())