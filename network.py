
from functools import partial
import pdb
import torch
import torch.nn as nn
from cnn_utils import conv_norm_lrelu,conv1x1
from vit_utils import *
from einops import rearrange
from pos_embed import get_2d_sincos_pos_embed
from utils import gradient,nolinear_trans_patch
from cnn_utils import res_conv_norm_lrelu
import time
class TransEX(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=10, in_chans=1,
                 embed_dim=512, step_depth=2, num_heads=8,
                 seq_len = 1000,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mode = "train"):
        super().__init__()


        self.seq_encoder = Seq_Trans_Encoder(in_chans=in_chans, seq_len=seq_len, embed_dim=embed_dim, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, step_depth=step_depth)
        

        self.patch_embed_map = PatchEmbed(250, 10, 32, 100)
        num_patches = self.patch_embed_map.num_patches

        self.pos_embed_map = nn.Parameter(torch.zeros(in_chans, num_patches+1, embed_dim),
                                         requires_grad=True)
        self.embed_dim = embed_dim


        self.cls_map = nn.Parameter(torch.zeros(1, 1, embed_dim))




        self.decoder_blocks = nn.Sequential(Mlp(in_features=512,hidden_features=2560,out_features=1988),
                                            nn.Sigmoid())

        self.in_chans = in_chans
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()



    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_map, std=.02)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):


            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, channel):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed_un.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], channel, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * channel ))
        return x

    def unpatchify(self, x,in_chans):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed_un.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, h * p))
        return imgs



    def forward_encoder(self, x, y):

        x = self.seq_encoder(x, y)


        return x

    def forward_decoder(self,latent):
        pred = self.decoder_blocks(latent)
        return pred

    def forward(self, seq, cmap):
        global_SUM = self.forward_encoder(seq, cmap)
        pred = self.forward_decoder(global_SUM)  # [N, L, p*p*3]
        return pred

class Cmap_Cnn_Encoder(nn.Module):
    def __init__(self,embed_dim=512,num_heads=16,mlp_ratio=4,norm_layer=nn.LayerNorm,step_depth=2):
        super().__init__()
        self.down = res_conv_norm_lrelu(input_dim=1, output_dim=8, mode='down')
        self.down2 = res_conv_norm_lrelu(input_dim=8, output_dim=16, mode='down')
        self.patch_embd = PatchEmbed(250,10,16,embed_dim)
        self.cls_cmap = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patchs = self.patch_embd.num_patches
        self.pos_embed_cmap = nn.Parameter(torch.zeros(1, self.num_patchs + 1, embed_dim),
                                          requires_grad=True)
        self.seq_en_stage1 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage2 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage3 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage4 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
    def forward(self, y):
        y = self.down(y)
        y = self.down2(y)
        # (10*10) -> 256
        y = self.patch_embd(y)
        y = y+self.pos_embed_cmap[:,1:,:]
        cls_cmap = self.cls_cmap + self.pos_embed_cmap[:, :1, :]
        cls_cmaps = cls_cmap.expand(y.shape[0], -1, -1)
        y = torch.cat([cls_cmaps,y],dim=1)
        y = self.seq_en_stage1(y)
        y = self.seq_en_stage2(y)
        y = self.seq_en_stage3(y)
        y = self.seq_en_stage4(y)
        return y[:, 0, :]

class Seq_Trans_Encoder(nn.Module):
    def __init__(self, in_chans=1, seq_len=1000, embed_dim=256, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm, step_depth=2):
        super().__init__()
        self.pos_embed_seq = nn.Parameter(torch.zeros(in_chans, seq_len + 1, embed_dim),
                                          requires_grad=True)
        self.embed_dim = embed_dim
        self.seq_embed = nn.Linear(in_features=21, out_features=embed_dim)
        self.cls_seq = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_en_stage1 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage2 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage3 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)
        self.seq_en_stage4 = stage_layer(embed_dim, num_heads, mlp_ratio, norm_layer, step_depth)

        self.cross_1 = cross_attn()
        self.cross_2 = cross_attn()
        self.cross_3 = cross_attn()

        torch.nn.init.normal_(self.cls_seq, std=.02)
    def forward(self,x,cmap):
        cmap = cmap.squeeze(1)
        x = self.seq_embed(x)
        x = x + self.pos_embed_seq[:, 1:, :]  # x的shape(2,1000,100)
        cls_seq = self.cls_seq + self.pos_embed_seq[:, :1, :]
        cls_seqs = cls_seq.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_seqs, x), dim=1)

        x = self.seq_en_stage1(x)
        x_cls = x[:, :1, :]
        x_feature = x[:, 1:, :]
        fuse = self.cross_1(x_feature, cmap)
        x = torch.cat([x_cls, fuse], dim=1)


        x = self.seq_en_stage2(x)
        x_cls = x[:, :1, :]
        x_feature = x[:, 1:, :]
        fuse = self.cross_2(x_feature, cmap)
        x = torch.cat([x_cls, fuse],dim=1)

        x = self.seq_en_stage3(x)
        x_cls = x[:, :1, :]
        x_feature = x[:, 1:, :]
        fuse = self.cross_3(x_feature, cmap)
        x = torch.cat([x_cls, fuse], dim=1)

        x = self.seq_en_stage4(x)
        
        
        return x[:, 0, :]


if __name__ == "__main__":
    y = torch.randn([1,1,1000,1000]).cpu()

    x = torch.randn([1,1000,21]).cpu()


    model = TransEX().cpu()
    pred = model(x, y)
    print(pred)
