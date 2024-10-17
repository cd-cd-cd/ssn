import torch
from torch import nn
import torch.nn.functional as F

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from model import SelfAttentionCell
from utils import _get_clones
from collections import OrderedDict
import math
from models.EfficientAdditiveAttnetion import EfficientAdditiveAttnetion
from models.ExternalAttention import ExternalAttention

# concat2crossAttention

clip_path = "/amax/home/chendian/huggingface/clip-32"
# clip_path = "/amax/home/chendian/.cache/clip/ViT-H-14/laionCLIP-ViT-H-14-laion2B-s32B-b79K"
class Model(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, args):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Model, self).__init__()

        self.img_tokens = 50

        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(clip_path)
        self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_path)
        self.processor = CLIPProcessor.from_pretrained(clip_path)

        self.clip_feature_dim = self.clip_vision_model.visual_projection.out_features
        self.clip_img_feature_dim = self.clip_vision_model.visual_projection.in_features
        self.projection_dim = args.projection_dim
        self.hidden_dim = self.projection_dim*2
        self.width = args.hidden_dim
        self.kl_weight = args.kl_weight

        # Image and Text embeddings
        self.token_type_embeddings = nn.Embedding(2, self.width)
        self.token_type_ref = 0
        self.token_type_text = 1

        # The Layer for degradation, to produce L+, I_r^0
        self.text_token_selection_mask = nn.Sequential(nn.Linear(self.width * 3, self.width),
                                                       nn.ReLU(),
                                                       nn.Linear(self.width, 1),
                                                       nn.Sigmoid())
        self.image_token_selection_mask = nn.Sequential(nn.Linear(self.width * 3, self.width),
                                                        nn.ReLU(),
                                                        nn.Linear(self.width, 1),
                                                        nn.Sigmoid())
        
        self.image_token_proj_layer = nn.Linear(self.clip_img_feature_dim, self.width)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, nhead=8)
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.combiner_layer = nn.Linear(self.projection_dim * 2, self.hidden_dim)
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                                          nn.Softplus(),
                                          nn.Dropout(0.5),
                                          nn.Linear(self.hidden_dim * 2, self.clip_feature_dim),
                                          nn.Softplus(),
                                          nn.Dropout(0.5),
                                          )
        self.dynamic_scalar = nn.Sequential(nn.Linear(self.projection_dim * 2, self.hidden_dim),
                                            nn.Softplus(),
                                            nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.Softplus(),
                                            nn.Linear(self.hidden_dim, self.hidden_dim),
                                            nn.Softplus(),
                                            nn.Linear(self.hidden_dim, 1), nn.Sigmoid()
                                            )
        self.logit_scale = 100
        self.loss = nn.CrossEntropyLoss()
        # cd edit combiner
        self.crossAttention = CrossAttention(self.clip_feature_dim, args.n_layers, args.n_heads, None)
        self.sa = SelfAttentionCell(args)
        self.Efficient = EfficientAdditiveAttnetion(in_dims=self.width, token_dim=512)
        self.External = ExternalAttention(d_model=self.width)
        
    def forward(self, reference_images: torch.tensor, text_inputs: torch.tensor,
                target_images: torch.tensor, ground_truth: torch.tensor) -> torch.tensor:
        reference_inputs = reference_images.squeeze()
        target_inputs = target_images.squeeze()

        # Extract the features with CLIP
        text_outputs = self.clip_text_model(**text_inputs)
        reference_outputs = self.clip_vision_model(reference_inputs)
        target_outputs = self.clip_vision_model(target_inputs)

        #### Degradation-Upgradation#####
        pos_fused_features, neg_fused_features = self.encode_features(
            reference_outputs.last_hidden_state, text_outputs, text_inputs['attention_mask'])
        target_features = self.encode_features(target_outputs.last_hidden_state, None, None)  # F_{tg} generation

        #### Linear Addition#####
        pos_predicted_features = self.combine_features(reference_outputs.image_embeds, text_outputs.text_embeds,
                                                   pos_fused_features[:, :self.img_tokens], pos_fused_features[:, self.img_tokens:],
                                                   text_mask=text_inputs['attention_mask'])
        neg_predicted_features = self.combine_features(reference_outputs.image_embeds, text_outputs.text_embeds,
                                                   neg_fused_features[:, :self.img_tokens], neg_fused_features[:, self.img_tokens:],
                                                   text_mask=text_inputs['attention_mask'])
        target_features = self.combine_features(target_outputs.image_embeds, None,
                                                   target_features[:, :self.img_tokens], None)

        logits = self.logit_scale * (torch.eye(len(reference_inputs), device=reference_inputs.device))
        pos_logits = self.logit_scale * pos_predicted_features @ target_features.T
        neg_logits = self.logit_scale * neg_predicted_features @ target_features.T

        loss_fct = nn.KLDivLoss()
        kl_loss = loss_fct(torch.log_softmax(pos_logits, dim=-1), torch.softmax(logits, dim=-1)) - \
                      loss_fct(torch.log_softmax(neg_logits, dim=-1), torch.softmax(pos_logits, dim=-1))

        loss_t = self.loss(pos_logits, ground_truth)
        loss_v = self.loss(pos_logits.T, ground_truth)
        loss = (loss_t + loss_v) / 2

        return loss + self.kl_weight * kl_loss

    def encode_image(self, images: torch.tensor):
        inputs = images.squeeze()
        outputs = self.clip_vision_model(inputs)
        return outputs

    def encode_text(self, text_inputs: torch.tensor):
        outputs = self.clip_text_model(**text_inputs)
        return outputs

    def pool_text(self, xx, text_mask):
        # average pool ignoring the masked entries
        xx = xx.transpose(-2, -1)
        embed_dim = xx.shape[1]
        # xx has shape: batch_size x feat_dim x num_tokens
        # to correctly broadcast, we must add a dimension to text_mask with unsqueeze
        xx = xx.masked_fill(text_mask.unsqueeze(1) == 0, 0.)
        denominators = (text_mask != 0).sum(1)
        denominators = denominators.float().unsqueeze(1)
        return torch.sum(xx, -1).view(-1, embed_dim) / denominators

    def token_type_embedding(self, ref_features, text_features, text_mask):
        temp = torch.ones(ref_features.shape[0], ref_features.shape[1], dtype=torch.long, device=ref_features.device)
        # 函数创建一个与 temp 形状相同的张量 ref_type_embedding，其所有元素都被填充为 self.token_type_ref 的值
        ref_type_embedding = torch.full_like(temp, self.token_type_ref) # 0
        ref_type_embedding = self.token_type_embeddings(ref_type_embedding)
        ref_embeddings = ref_features + ref_type_embedding

        if text_features is not None:
            text_type_embedding = torch.full_like(text_mask, self.token_type_text) # 1
            text_type_embedding = self.token_type_embeddings(text_type_embedding.long())
            text_embeddings = text_features + text_type_embedding
            return ref_embeddings, text_embeddings
        else:
            return ref_embeddings
    
    # raw
    def combine_features(self, reference_embeds, text_embeds, reference_features, text_features, text_mask= None) -> torch.tensor:
        cls_ref_embeds = reference_features[:, 0]
    
        cls_ref_embeds_sa = self.sa(cls_ref_embeds.unsqueeze(0)).squeeze(0)
        if text_features is not None:
            cls_text_embeds = self.pool_text(text_features, text_mask)
            
            raw_combined_features = torch.cat((cls_text_embeds, cls_ref_embeds), -1)
            combined_features = F.relu(self.combiner_layer(raw_combined_features))
            dynamic_scalar = self.dynamic_scalar(raw_combined_features)
            output = 0.01 * self.output_layer(combined_features) + dynamic_scalar * text_embeds + (
                1 - dynamic_scalar) * reference_embeds
            
            
            cls_text_embeds_sa = self.sa(cls_text_embeds.unsqueeze(0)).squeeze(0)
            raw_combined_features_sa = torch.cat((cls_ref_embeds_sa, cls_text_embeds_sa), -1)
            combined_features_sa = F.relu(self.combiner_layer(raw_combined_features_sa))
            dynamic_scalar = self.dynamic_scalar(combined_features_sa)
            output_sa = 0.01 * self.output_layer(combined_features_sa) + dynamic_scalar * text_embeds + (
                1 - dynamic_scalar) * reference_embeds
            output = 0.8 * output + 0.2 * output_sa
        else:
            output = 0.01 * cls_ref_embeds + reference_embeds

        return F.normalize(output, dim=-1)


    def encode_features(self, image_features: torch.tensor, text_outputs, text_mask) -> torch.tensor:
        if text_outputs is not None:
            text_features = text_outputs.last_hidden_state
            image_projected_features = self.image_token_proj_layer(image_features)

            global_img = image_projected_features[:, 0].unsqueeze(1)

            ##### Decompose text instructions ###########
            B, N, C = text_features.size()
            local_x = text_features
            global_img = global_img.expand(B, N, C)
            cross_feat = self.crossAttention(local_x, global_img)
            sa_feat = self.sa(cross_feat)
            cross_feat = self.crossAttention(sa_feat, cross_feat)
            fusion_feat = torch.cat([cross_feat, local_x, global_img], dim=-1)
            
            
            selection_mask = self.text_token_selection_mask(fusion_feat)

            positive_text_features = text_features * selection_mask.expand_as(text_features)
            negative_text_features = text_features * (1 - selection_mask).expand_as(text_features)

            ##### To obtain visual prototypes ###########
            global_x = self.pool_text(negative_text_features, text_mask).unsqueeze(1)
            B, N, C = image_projected_features.size()
            local_x = image_projected_features
            global_x = global_x.expand(B, N, C)
            cross_feat = self.crossAttention(local_x, global_x)
            sa_feat = self.sa(cross_feat)
            cross_feat = self.crossAttention(sa_feat, cross_feat)
            
            fusion_feat = torch.cat([cross_feat, local_x, global_x], dim=-1)
            selection_mask = self.image_token_selection_mask(fusion_feat)
        

            positive_image_features = image_projected_features * selection_mask.expand_as(image_projected_features)
            negative_image_features = image_projected_features * (1 - selection_mask).expand_as(image_projected_features)
            visual_prototypes = positive_image_features

            positive_image_features, positive_text_features = self.token_type_embedding(
                positive_image_features, positive_text_features, text_mask)
            negative_image_features, negative_text_features = self.token_type_embedding(
                negative_image_features, negative_text_features, text_mask)
            
            ##### fusion decomposed reference image features and text features by fusion layer ######
            co_features = torch.cat((positive_image_features, positive_text_features), dim=1)
            # The input for nn.Transformers is size of (seq_len, bs,  dim)
            co_features = co_features.permute(1, 0, 2)
            pos_fused_features = self.fusion_layer(co_features)
            pos_fused_features = pos_fused_features.permute(1, 0, 2)

            co_features = torch.cat((negative_image_features, negative_text_features), dim=1)
            # The input for nn.Transformers is size of (seq_len, bs,  dim)
            co_features = co_features.permute(1, 0, 2)
            neg_fused_features = self.fusion_layer(co_features)
            neg_fused_features = neg_fused_features.permute(1, 0, 2)

            return pos_fused_features, neg_fused_features
        else:
            image_projected_features = self.image_token_proj_layer(image_features)
            image_projected_features= self.token_type_embedding(
                image_projected_features, None, None)
            
            co_features = image_projected_features
            # The input for nn.Transformers is size of (seq_len, bs, dim)
            co_features = co_features.permute(1, 0, 2)
            fused_features = self.fusion_layer(co_features)
            fused_features = fused_features.permute(1, 0, 2)

            return fused_features
    
    
    def extract_query(self, textual_query, visual_query):
        textual_query = F.normalize(self.extract_text_fea(textual_query), p=2, dim=-1)
        visual_query = F.normalize(self.extract_img_fea(visual_query), p=2, dim=-1)
        
        combined_feature = self.combiner_fc(torch.cat([textual_query, visual_query], dim=-1))
        dynamic_scaler = self.scaler_fc(self.dropout(combined_feature))
        query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query
        return F.normalize(query, p=2, dim=-1)
    
class CrossAttention(nn.Module):
    def __init__(self, clip_feature_dim, n_layers, n_heads, attn_mask=None):
        super(CrossAttention, self).__init__()
        self.n_layers = n_layers
        self.resblocks = _get_clones(ResidualCrossAttentionBlock(clip_feature_dim, n_heads, attn_mask), n_layers)

    def forward(self, x, y):
        for i in range(self.n_layers):
            x = self.resblocks[i](x, y)
        return x
    
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super(ResidualCrossAttentionBlock, self).__init__()

        self.attn = CrossAttentionLayer(d_model, n_head)
        self.ln_x1 = nn.LayerNorm(d_model)
        self.ln_y1 = nn.LayerNorm(d_model)
        self.mlp_ratio = 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * self.mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(d_model * self.mlp_ratio), d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, y):
        return self.attn(x, y)
        
    def forward(self, x, y):
        x = x + self.attention(self.ln_x1(x), self.ln_y1(y))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super(CrossAttentionLayer, self).__init__()
        self.h = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.projections = _get_clones(nn.Linear(d_model, d_model), 3)

    def forward(self, x, y):
        nbatches = x.size(0)
        query, key, value = [l(v).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, v in zip(self.projections, (y, x, x))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return x
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
