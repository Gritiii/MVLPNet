import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        
from model.few_seg.clip import clip
import numpy as np
import random
import time
from pytorch_grad_cam import GradCAM
import cv2
import math
from model.few_seg.TVCE import get_img_cam
from model.backbone.layer_extrator import layer_extrator
from einops import rearrange
from .loss import WeightedDiceLoss
from model.util.ASPP import ASPP, ASPP_Drop ,ASPP_BN
from model.util.PSPNet import OneModel as PSPNet
from torch.cuda.amp import autocast as autocast
from model.few_seg.clip_text import new_class_names,isaid_class_names


class OT(nn.Module):

    def __init__(self, dim_v, dim_t, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out
        self.scale = qk_scale or dim_out ** -0.5
        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.k_proj_pre_1 = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.v_proj_pre_1 = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.proj_post_t = nn.Conv1d(dim_out, dim_out, kernel_size=1)
        self.prompt_temp_l1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.beta_t = 1
        self.eps = 0.05  # Sinkhorn regularization parameter
        self._initialize_weights()
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def Sinkhorn_log_exp_sum(self, C, mu, nu, epsilon):
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        thresh = 1e-6
        max_iter = 100
        for i in range(max_iter):
            u0 = u
            K = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
            K /= epsilon
            u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=2)
            u = epsilon * u_ + u
            K_t = K.permute(0, 2, 1).contiguous()
            v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=2)
            v = epsilon * v_ + v
            err = (u - u0).abs().mean()
            if err.item() < thresh:
                break
        K = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
        K /= epsilon
        T = torch.exp(K)
        return T

    def forward(self, F_t, F_s):
        B1, N1, C1 = F_t.shape
        B2, N2, C2 = F_s.shape
        assert B1 == B2  
        self.q_proj_pre = self.q_proj_pre.to(F_t.dtype)
        self.k_proj_pre_1 = self.k_proj_pre_1.to(F_s.dtype)
        self.v_proj_pre_1 = self.v_proj_pre_1.to(F_s.dtype)
        self.proj_post_t = self.proj_post_t.to(F_t.dtype)
        q_t = self.q_proj_pre(F_t.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)
        k_s = self.k_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)
        v_s = self.v_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)
        q_t = F.normalize(q_t, dim=-1, p=2)
        k_s = F.normalize(k_s, dim=-1, p=2)
        sim = torch.einsum('bnkc,bmkc->bknm', q_t, k_s) * self.beta_t  # (B, num_heads, N1, N2)
        wdist = 1.0 - sim
        wdist = wdist.view(B1 * self.num_heads, N1, N2)
        sim = sim.view(B1 * self.num_heads, N1, N2)  
        xx = torch.ones(B1 * self.num_heads, N1, dtype=sim.dtype, device=sim.device) / N1
        yy = torch.ones(B1 * self.num_heads, N2, dtype=sim.dtype, device=sim.device) / N2
        T = self.Sinkhorn_log_exp_sum(wdist, xx, yy, self.eps)  # Optimal transport matrix
        score_map = (N1 * N2 * sim * T).view(B1, self.num_heads, N1, N2)
        score_map = self.attn_drop(score_map)
        F_t_a = torch.einsum('bknm,bmkc->bnkc', score_map, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = self.proj_post_t(F_t_a.permute(0, 2, 1)).permute(0, 2, 1)
        F_t_a = F_t_a / F_t_a.norm(dim=-1, keepdim=True)
        F_t_a = self.proj_drop(F_t_a)  # Normalize the output
        return F_t_a
def SA_Weighted_GAP(supp_feat, mask, supp_pred_mask):
    supp_pred = supp_pred_mask+mask
    new_mask1 = torch.zeros_like(mask)
    new_mask2 = torch.zeros_like(mask)

    new_mask1[supp_pred==2] = 1
    new_mask2[supp_pred==1] = 1

    new_mask1[mask==0] = 0
    new_mask2[mask==0] = 0

    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    new_area1 = F.avg_pool2d(new_mask1, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    new_area2 = F.avg_pool2d(new_mask2, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat1 = supp_feat * new_mask1
    supp_feat1 = F.avg_pool2d(input=supp_feat1, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area1
    supp_feat2 = supp_feat * new_mask2
    supp_feat2 = F.avg_pool2d(input=supp_feat2, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / new_area2
    return supp_feat1, supp_feat2



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class PGSA(nn.Module):

    def __init__(self, embed_dims, drop=0.):
        super(PGSA, self).__init__()
        self.embed_dims = embed_dims
        self.drop = drop
        
        self.norm1 = nn.LayerNorm(embed_dims)

        self.q = nn.Linear(embed_dims, embed_dims, bias=False)
        self.k = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v = nn.Linear(embed_dims, embed_dims, bias=False)
        self.proj_x = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.proj_y = nn.Linear(embed_dims * 2, embed_dims, bias=False)

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn_x = Mlp(in_features=embed_dims, hidden_features=embed_dims, act_layer=nn.GELU, drop=drop)
        self.res1 = nn.Sequential(
            nn.Conv2d(embed_dims*3, embed_dims, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(embed_dims*3, embed_dims, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )
        self.cls = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(embed_dims, 2, kernel_size=1)
        )             



    def forward(self, x, y, shape, mask=None):
        b, _, c = x.size()
        h, w = shape
        ##x是查询   y是支持
        # skip connection
        x_skip = x  # b, n, c
        y_skip = y  # b, n, c
        
        # reshape
        mask = mask.view(b, h, w, 1).permute(0, 3, 1, 2).contiguous()  # b, 1, h, w
        
        # layer norm
        x = self.norm1(x)
        y = self.norm1(y)
        
        # input projection
        q = self.q(y)  # Support: b, n, c
        k = self.k(x)  # Query: b, n, c 16,1024,256
        v = self.v(x)  # Query: b, n, c
        
        # prototype extraction
        q = q.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        fg_pro = Weighted_GAP(q, mask)#16,256,1,1
        fg_pro1 = fg_pro.expand_as(q).repeat(1,2,1,1)#16,256×2,32,32
        fg_pro1 = torch.cat([q,fg_pro1], dim = 1)#16,256×3,32,32
        fg_pro1 = self.res1(fg_pro1)#16,256,32,32
        fg_pro1 = self.res2(fg_pro1)+fg_pro1#16,256,32,32
        out = self.cls(fg_pro1)#16,2,32,32
        supp_pred_mask = torch.argmax(out, dim=1, keepdim=True)
        supp_feat1, supp_feat2 = SA_Weighted_GAP(q, mask, supp_pred_mask)#q:16,256,32,32.mask:16,1,32,32.supp_pred_mask:16,1,32,32
        supp_feat1 = supp_feat1.expand_as(q)##16,256,32,32
        supp_feat2 = supp_feat2.expand_as(q)#16,256,32,32
        q_xiu = torch.cat([supp_feat1,supp_feat2,q],dim = 1)#16,256×3,32,32
        q_xiu = self.res3(q_xiu)
        fg_pro = Weighted_GAP(q_xiu, mask).squeeze(-1)  # b, c, 1   16,256,1,1----16,256,1
        bg_pro = Weighted_GAP(q_xiu, 1 - mask).squeeze(-1)  # b, c, 1 16,256,1    
        # normalize
        cosine_eps = 1e-7
        k_norm = torch.norm(k, 2, 2, True) #16,1024,1
        fg_pro_norm = torch.norm(fg_pro, 2, 1, True) #16,1,1
        bg_pro_norm = torch.norm(bg_pro, 2, 1, True)
        
        # cosine similarity
        fg_scores = torch.einsum("bmc,bcn->bmn", k, fg_pro) / (torch.einsum("bmc,bcn->bmn", k_norm, fg_pro_norm) + cosine_eps)
        bg_scores = torch.einsum("bmc,bcn->bmn", k, bg_pro) / (torch.einsum("bmc,bcn->bmn", k_norm, bg_pro_norm) + cosine_eps)#过滤掉与支持背景相似的查询特征
        
        # normalization
        fg_scores = fg_scores.squeeze(-1)  # b, n
        print(fg_scores.shape)
        bg_scores = bg_scores.squeeze(-1)
        print(bg_scores.shape)  # b, n
        
        fg_scores = (fg_scores - fg_scores.min(1)[0].unsqueeze(1)) / (
            fg_scores.max(1)[0].unsqueeze(1) - fg_scores.min(1)[0].unsqueeze(1) + cosine_eps)
        bg_scores = (bg_scores - bg_scores.min(1)[0].unsqueeze(1)) / (
            bg_scores.max(1)[0].unsqueeze(1) - bg_scores.min(1)[0].unsqueeze(1) + cosine_eps)
        
        fg_scores = fg_scores.unsqueeze(-1)
        bg_scores = bg_scores.unsqueeze(-1)
        
        # discriminative region
        scores = fg_scores - bg_scores  # b, n, 1
        pseudo_mask = scores.clone().permute(0, 2, 1).contiguous()
        pseudo_mask = pseudo_mask.view(b, 1, *shape)  # b, 1, h, w
        
        # truncate score
        score_mask = torch.zeros_like(scores)  # b, n, 1
        score_mask[scores < 0] = -100.
        scores = scores + score_mask  # b, n, 1
        
        # softmax
        scores = F.softmax(scores, dim=1)
        
        # output
        query_pro = scores.transpose(-2, -1).contiguous() @ v  # b, 1, c 获得判别性的查询前景特征
        
        # similarity-based prototype fusion
        query_pro_norm = torch.norm(query_pro, 2, 2, True)  # b, 1, c
        sim = torch.einsum("bmc,bcn->bmn", query_pro, fg_pro) / (torch.einsum("bmc,bcn->bmn", query_pro_norm, fg_pro_norm) + cosine_eps)#高度相似的区域
        sim = (sim + 1.) / 2.  # b, 1, 1
        pro = sim * fg_pro.transpose(-2, -1).contiguous() + (1. - sim) * query_pro  # b, 1, c 公共区域挖掘和自挖掘
    
        # projection
        x = x_skip + self.proj_x(torch.cat([x_skip, pro.expand_as(x_skip)], dim=-1))

        # ffn    
        x = x + self.ffn_x(self.norm2(x))
        return x, pseudo_mask



def zeroshot_classifier(classnames, templates, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()




def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    
    fea_T = fea.permute(0, 2, 1)    
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    
    return gram

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.shot = args.shot
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion_dice = WeightedDiceLoss()
        self.pretrained = args.pretrain
        self.classes = 2
        self.fp16 = args.fp16
        self.backbone = args.backbone
        self.base_class_num = args.base_class_num
        self.clip_model, _ = clip.load('/data6/zhenhaoyang/FSS/R2Net-main/initmodel/clip/ViT-B-16.pt')  
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual.transformer.resblocks[-1].parameters():
            param.requires_grad = True
        self.bg_text_features = zeroshot_classifier(isaid_class_names, ['a photo without {}.'],
                                                        self.clip_model)
        self.fg_text_features = zeroshot_classifier(isaid_class_names, ['a photo of {}.'],
                                                        self.clip_model)
        self.PGSA = PGSA(embed_dims = 256, drop=0.)
        self.OT = OT(dim_v = 512,dim_t = 512, dim_out = 512)




        if self.pretrained:
            BaseNet = PSPNet(args)
            weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.dataset, args.split, args.backbone)
            new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            print('load <base> weights from: {}'.format(weight_path))
            try: 
                BaseNet.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                BaseNet.load_state_dict(new_param)
            
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = BaseNet.layer0, BaseNet.layer1, BaseNet.layer2, BaseNet.layer3, BaseNet.layer4

            self.base_layer = nn.Sequential(BaseNet.ppm, BaseNet.cls[0], BaseNet.cls[1])
            self.base_learner =  nn.Sequential(BaseNet.cls[2], BaseNet.cls[3], BaseNet.cls[4])
        else:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone, pretrained=True)

        reduce_dim = 256
        if self.backbone == 'vgg':
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  
        channel = 4

        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + channel, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=1, padding=0, bias=False)
        )
        self.annotation_root = "/data6/zhenhaoyang/FSS/data/iSAID/ann_dir"

        depths = (8,)
        scale = 0
        for i in range(len(depths)):
            scale += 2 ** i
        self.ASPP_meta = ASPP(scale * reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(scale * reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )    
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )    
        self.relu = nn.ReLU(inplace=True)
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))
        if args.shot > 1:
            self.kshot_trans_dim = 2
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))



    def get_optim(self, model, args, LR):
        params = [
        {'params': model.down_query.parameters()},
        {'params': model.down_supp.parameters()},
        {'params': model.init_merge_query.parameters()},
        {'params': model.init_merge_supp.parameters()},
        {'params': model.ASPP_meta.parameters()},
        {'params': model.res1_meta.parameters()},
        {'params': model.res2_meta.parameters()},
        {'params': model.cls_meta.parameters()},
        {'params': model.gram_merge.parameters()},
        {'params': model.cls_merge.parameters()},
        {'params': model.OT.parameters()},
        {'params': model.PGSA.parameters()},
        
     ]
        if self.shot > 1:
            params.append({'params': model.kshot_rw.parameters()})
    
        optimizer = torch.optim.SGD(params, lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
    
        return optimizer



    def forward(self, x, x_cv2,que_name,class_name,s_x, s_y, y, cat_idx=None):
            x_size = x.size()
            bs = x_size[0]
            h = x_size[2]
            w = x_size[3]
            s_x = rearrange(s_x, "b n c h w -> (b n) c h w")

            # Query Feature
            with torch.no_grad():
                query_feat_0 = self.layer0(x)
                query_feat_1 = self.layer1(query_feat_0)
                query_feat_2 = self.layer2(query_feat_1)#[8,512,64,64]
                query_feat_3 = self.layer3(query_feat_2)#[8,1024,64,64]
                query_feat_4 = self.layer4(query_feat_3)
                query_out = self.base_layer(query_feat_4)
            
                supp_feat_0 = self.layer0(s_x)
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4_true = self.layer4(supp_feat_3)    
                supp_base_out = self.base_layer(supp_feat_4_true.clone())            

            if self.backbone == 'vgg':
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            query_feat_high_4 = query_feat_4#[8,2048,64,64]
            query_feat_high_5 = query_out#[8,512,64,64]
            query_feat = torch.cat([query_feat_3, query_feat_2], 1)#[8,1536,64,64]
            fts_size = query_feat.size()[-2:]
            supp_feat = torch.cat([supp_feat_2, supp_feat_3], dim=1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat_mid = supp_feat.view(bs, self.shot, -1, fts_size[0], fts_size[1])
            supp_feat_high_4 = supp_feat_4_true.view(bs, self.shot, -1, fts_size[0], fts_size[1])
            supp_feat_high_5 = supp_base_out.view(bs, self.shot, -1, fts_size[0], fts_size[1])

            query_feat = self.down_query(query_feat)
            print("000")
            print(query_feat.shape)
            # Support Feature     
            final_supp_list_4 = []
            final_supp_list_5 = []
            target_layers = [self.clip_model.visual.transformer.resblocks[-1].ln_1]
            cam = GradCAM(model=self.clip_model, target_layers=target_layers, reshape_transform=reshape_transform)
            img_cam_list  = get_img_cam(x_cv2, que_name, class_name, self.clip_model, self.bg_text_features, self.fg_text_features, cam, self.annotation_root, self.OT,self.training)
            img_cam_list = [F.interpolate(t_img_cam.unsqueeze(0).unsqueeze(0), size=(supp_feat.shape[2], supp_feat.shape[3]), mode='bilinear',
                                        align_corners=True) for t_img_cam in img_cam_list]
            img_cam = torch.cat(img_cam_list, 0)
            img_cam = img_cam.repeat(1,2,1,1)



            mask_list = []
            supp_pro_list = []

            supp_feat_list = []

            for i in range(self.shot):
                mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
                mask = F.interpolate(mask, size=fts_size, mode='bilinear', align_corners=True)
                mask_list.append(mask)
                final_supp_list_4.append(supp_feat_high_4[:, i, :, :, :])
                final_supp_list_5.append(supp_feat_high_5[:, i, :, :, :])
                supp_feat_list.append((supp_feat_mid[:, i, :, :, :]).unsqueeze(-1))
                supp_pro = Weighted_GAP(supp_feat_mid[:, i, :, :, :], mask)
                supp_pro_list.append(supp_pro)  #中层特征的支持原型 
            supp_mask = torch.cat(mask_list, dim=1).mean(1, True)
            supp_feat = torch.cat(supp_feat_list, dim=-1).mean(-1)  # bs, 256, 64, 64
            supp_pro = torch.cat(supp_pro_list, dim=2).mean(2, True) 
            supp_pro = supp_pro.expand_as(query_feat)
            corr_fg_4, _, corr_4, corr_fg_4_sim_max, corr_4_sim_max = self.generate_prior_proto(query_feat_high_4, final_supp_list_4, mask_list, fts_size)
            corr_fg_5, _, corr_5, corr_fg_5_sim_max, corr_5_sim_max = self.generate_prior_proto(query_feat_high_5, final_supp_list_5, mask_list, fts_size)
                # gen pro
            corr_fg = corr_fg_4.clone()
            corr = corr_4.clone() 
            for i in range(bs):
                for j in range(self.shot):
                    if corr_fg_4_sim_max[i, j] < corr_fg_5_sim_max[i, j]:
                        corr_fg[i, j] = corr_fg_5[i, j]
                    if corr_4_sim_max[i, j] < corr_5_sim_max[i, j]:
                        corr[i, j] = corr_5[i, j]
            corr_fg = corr_fg.mean(1, True)
            corr = corr.mean(1, True)
            corr_query_mask = torch.cat([corr_fg, corr], dim=1)
            print("111")
            print(corr_query_mask.shape)
            query_cat = torch.cat([query_feat, supp_pro, corr_query_mask,img_cam * 10], dim=1)  
            query_feat = self.init_merge_query(query_cat) 
            print("222")
            print(query_feat.shape)
            supp_cat = torch.cat([supp_feat, supp_pro, supp_mask], dim=1)  
            supp_feat = self.init_merge_supp(supp_cat)
            print("333")
            print(supp_feat.shape)
            Wh, Ww = query_feat.size(2), query_feat.size(3)
            q = query_feat.flatten(2).transpose(1, 2)  # bs, hw, c
            s = supp_feat.flatten(2).transpose(1, 2)  # bs, hw, c
            s_mask = supp_mask.flatten(2).transpose(1, 2)   

            q, pseudo_mask = self.PGSA(q, s, (Wh, Ww), s_mask)
            out = q.view(-1, Wh, Ww, 256).permute(0, 3, 1, 2).contiguous()
       
            query_meta = self.ASPP_meta(out)
            query_meta = self.res1_meta(query_meta)
            query_meta = self.res2_meta(query_meta) + query_meta
            meta_out = self.cls_meta(query_meta)
            base_out = self.base_learner(query_feat_high_5)
            meta_out_soft = meta_out.softmax(1)
            base_out_soft = base_out.softmax(1)
            bs = x.shape[0]
            que_gram = get_gram_matrix(query_feat_2)
            norm_max = torch.ones_like(que_gram).norm(dim=(1,2))
            est_val_list = []
            supp_feat_list = rearrange(supp_feat_2, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_list = [supp_feat_list[:, i, ...] for i in range(self.shot)]
            for supp_item in supp_feat_list:
                supp_gram = get_gram_matrix(supp_item)
                gram_diff = que_gram - supp_gram
                est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1)) # norm2
            est_val_total = torch.cat(est_val_list, 1) 
            if self.shot > 1:
                val1, idx1 = est_val_total.sort(1)
                val2, idx2 = idx1.sort(1)
                weight = self.kshot_rw(val1)
                idx3 = idx1.gather(1, idx2)
                weight = weight.gather(1, idx3)
                weight_soft = torch.softmax(weight, 1)
            else:
                weight_soft = torch.ones_like(est_val_total)
            est_val = (weight_soft * est_val_total).sum(1, True)
            meta_map_bg = meta_out_soft[:, 0:1, :, :]                           
            meta_map_fg = meta_out_soft[:, 1:, :, :]  
            if self.training:
                c_id_array = torch.arange(self.base_class_num + 1, device='cuda')
                base_map_list = []
                for b_id in range(bs):
                    c_id = cat_idx[0][b_id] + 1
                    c_mask = (c_id_array != 0) & (c_id_array != c_id)
                    base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
                base_map = torch.cat(base_map_list,0)
            else:
                base_map = base_out_soft[:, 1:, :, :].sum(1, True)
            est_map = est_val.expand_as(meta_map_fg)
            meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
            meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))
            merge_map = torch.cat([meta_map_bg, base_map], dim=1)
            merge_bg = self.cls_merge(merge_map)  
            final_out = torch.cat([merge_bg, meta_map_fg], dim=1) 
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)
            if self.training:
                main_loss = self.criterion_ce(final_out, y.long())
                aux_loss2 = torch.zeros_like(main_loss)
                aux_loss1 = self.criterion_dice(meta_out, y.long())
                gt_mask = (y == 1).float().unsqueeze(1)
                gt_mask = F.interpolate(gt_mask, size=pseudo_mask.size()[-2:], mode='nearest')
                aux_loss2 += self.criterion_bce(pseudo_mask, gt_mask.long().float())
                alpha = 0.2

                return final_out.max(1)[1], main_loss,aux_loss1,alpha * aux_loss2 
            else:
                return final_out









    def cos_sim(self, query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        return similarity

    def generate_prior_proto(self, query_feat_high, final_supp_list, mask_list, fts_size):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        fg_list = []
        bg_list = []
        fg_sim_maxs = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)#支持前景的原型
            bg_supp_feat = Weighted_GAP(tmp_supp_feat, 1 - tmp_mask)#支持背景的原型

            fg_sim = self.cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
            bg_sim = self.cos_sim(query_feat_high, bg_supp_feat, cosine_eps)

            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            
            fg_sim_max = fg_sim.max(1)[0]  # bsize
            fg_sim_maxs.append(fg_sim_max.unsqueeze(-1))  # bsize, 1

            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                        fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                    bg_sim.max(1)[0].unsqueeze(1) - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)##归一化

            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)

            fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
            bg_sim = F.interpolate(bg_sim, size=fts_size, mode='bilinear', align_corners=True)
            fg_list.append(fg_sim)
            bg_list.append(bg_sim)
        fg_corr = torch.cat(fg_list, 1)  # bsize, shots, h, w
        bg_corr = torch.cat(bg_list, 1)
        corr = (fg_corr - bg_corr)
        corr[corr < 0] = 0
        corr_max = corr.view(bsize, len(final_supp_list), -1).max(-1)[0]  # bsize, shots
        
        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)  # bsize, shots
        return fg_corr, bg_corr, corr, fg_sim_maxs, corr_max
