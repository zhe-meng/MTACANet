import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from typing import Tuple
from einops import rearrange

from WTConv import WTConv
from FCAttention import Attention


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)




class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)) #[-0.6931, -0.1945, -0.0645, -0.0223] [-0.6931, -0.0925, -0.0157, -0.0028]
        self.register_buffer('angle', angle) 
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay) # tensor([-6.9315e-01, -3.7871e-02, -2.7660e-03, -2.0530e-04])
        index_w = torch.arange(W).to(self.decay) # tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
        grid = torch.meshgrid([index_h, index_w]) # 
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2) 衰减掩码的大小为 (H*W, H*W)，表示图像中每个像素与其他所有像素的关系
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        # mask = (mask.abs()).sum(dim=-1) # 曼哈顿距离为坐标值之差取绝对值 求和
        mask = torch.sqrt((mask ** 2).sum(dim=-1)) #欧式距离 为坐标值之差先平方求和，再开根号
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)

        return mask

    # def generate_2d_decay(self, H: int, W: int):
    #     '''
    #     generate 2d decay mask, the result is (HW)*(HW)
    #     '''
    #     index_h = torch.arange(H).to(self.decay)
    #     index_w = torch.arange(W).to(self.decay)
    #     grid = torch.meshgrid([index_h, index_w])
    #     grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2) 衰减掩码的大小为 (H*W, H*W)，表示图像中每个像素与其他所有像素的关系
    #     mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
    #     mask = (mask.abs()).sum(dim=-1)
    #     mask = mask * self.decay[:, None, None]  #(n H*W H*W)


    #     return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None]  #(n l l)


        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)  #h*w
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1) [13, 13, 16]
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l) [4, 169, 169]
            
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
    

class MaSAd(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w * mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h * mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
class MaSA(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2) 
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x) 
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1) 
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)
        qr = qr.flatten(2, 3) #(b n l d1) 

        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l) 

        qk_mat = qk_mat * mask  #(b n l l)
        
        
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output#, qk_mat



class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=False
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
    







    
class RetBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., chunkwise_recurrent=False ,layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        
        if chunkwise_recurrent:
            self.retention = MaSAd(embed_dim, num_heads)
        else:
            self.retention = MaSA(embed_dim, num_heads)
        
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
       
        

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):

        x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))#
        x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x




class MTACANet(nn.Module):

    def __init__(self, in_chans=200, num_classes=1000, embed_dim=96, num_heads=8, chunkwise_recurrent=False, ffn_dim=96, F_kernel_size=1, F_dim=15):
        super().__init__()
       
       
        self.chunkwise_recurrent = chunkwise_recurrent
              
        self.wt = WTConv(in_chans, embed_dim)

        
        # MSPA
        self.dropout = nn.Dropout(0.3)
        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels=embed_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.conv2d_channel = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels=embed_dim, kernel_size=(1, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.conv2d_fusion = nn.Sequential(
            nn.Conv2d(4 * embed_dim, out_channels=embed_dim, kernel_size=(1, 1)),
            nn.GELU(),
        )


        # MSPA
        self.spe4 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels=embed_dim, kernel_size=(1, 1)),
            nn.GELU(),
        )
        self.spe_token1 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3)),
            # nn.BatchNorm2d(1),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.spe_token2 = nn.Sequential(
            nn.Conv3d(1, 1, (1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            # nn.BatchNorm2d(1),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )   

        self.AGLCA = Attention(channel=embed_dim)

        self.Relpos = RelPos2d(embed_dim, num_heads, F_kernel_size, F_dim)
        self.blocks = RetBlock(embed_dim, num_heads, ffn_dim)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        self.norm = nn.BatchNorm2d(embed_dim)

       


    def forward(self, x):  #[64, 1, 64, 13, 13]
        x = x.squeeze(dim=1)
       
        #WTConv
        x = self.wt(x)  #[64, 64, 13, 13]
         

        # MSPE
        x_e = x.permute(0, 2, 3, 1) #[64, 13, 13, 64]
        x_e = x_e.unsqueeze(1)  ##[64, 1, 13, 13, 64]
        x_e1 = self.spe_token1(x_e) #[64, 1, 13, 13, 64]
        x_e2 = self.spe_token2(x_e) #[64, 1, 13, 13, 64]
        x_e = x_e1 + x_e2 + x_e #[64, 1, 13, 13, 64]
        x_e = self.dropout(x_e).squeeze(1).permute(0, 3, 1, 2)  #([64, 64, 13, 13]
 

        #AGLCA
        x_ca = self.AGLCA(x_e) #[64, 64, 13, 13]
        x_ca = x_ca.permute(0, 2, 3, 1) #[64, 13, 13, 64]
       

        # MSPA      
        x1 = self.conv2d_channel(x) #[64, 64, 13, 13]
        x2 = self.conv2d_features1(x) #[64, 64, 13, 13]
        x3 = self.conv2d_features2(x) #[64, 64, 13, 13]
        x = torch.cat([x, x1, x2, x3], dim=1) #[64, 256, 13, 13]
        x = self.conv2d_fusion(x)  #[64, 64, 13, 13]
        x_a = self.dropout(x) #[64, 64, 13, 13]
        x_a= x_a.permute(0, 2, 3, 1) #[64, 64, 13, 13]
       

        #MaSA
        b, h, w, d = x_a.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=False)
        x_masa = self.blocks(x_a, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)#[64, 13, 13, 64]
        
        x = x_ca + x_masa
     
        
        x = self.norm(x.permute(0, 3, 1, 2)).flatten(2, 3) #[64, 64, 169]
        x = self.avgpool(x)   #[64, 64, 1]
        x = torch.flatten(x, 1) #[64, 64]
        x = self.head(x) #[64, 16]
        
        return x
    


if __name__ == '__main__':

    model = MTACANet(in_chans=64, num_classes=16,embed_dim=64, num_heads=4,ffn_dim=96)
    model.eval()
    #print(model)
    
   
   
    input = torch.randn(100, 1, 64, 13, 13)
    
    
    y = model(input)
    
    print(model)
    print(y.size())

