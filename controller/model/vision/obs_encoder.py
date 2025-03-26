import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops.layers.torch import Rearrange

from controller.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

feature_dim_dict = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536
}

class ObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_config: dict,
        ):
        super().__init__()

        if model_config['head']['local_weights_path'] is None:
            self.dino_head = torch.hub.load(
                "facebookresearch/dinov2",
                model_config['head']['model_type'],
                pretrained=True,
                force_reload=False,
                trust_repo=True
            )
        else:
            self.dino_head = torch.hub.load(
                repo_or_dir=model_config['head']['local_weights_path'],  
                model=model_config['head']['model_type'],
                source='local',
            )

        self.dino_head.eval()

        if model_config['wrist']['local_weights_path'] is None:
            self.dino_wrist = torch.hub.load(
                "facebookresearch/dinov2",
                model_config['wrist']['model_type'],
                pretrained=True,
                force_reload=False,
                trust_repo=True
            )
        else:
            self.dino_wrist = torch.hub.load(
                repo_or_dir=model_config['wrist']['local_weights_path'],  
                model=model_config['wrist']['model_type'],
                source='local',
            )
            # self.dino_wrist = torch.hub.load(
            #     "facebookresearch/dinov2",
            #     model_config['wrist']['model_type'],
            #     pretrained=False
            # )
            # self.dino_wrist.load_state_dict(torch.load(model_config['wrist']['local_weights_path']))

        self.dino_wrist.eval()

        # Set all parameters to not require grad
        for param in self.dino_head.parameters():
            param.requires_grad = False
        for param in self.dino_wrist.parameters():
            param.requires_grad = False
        # 用于对输入图像色彩增强 随机调整图像的颜色属性（亮度、对比度、饱和度和色调 提高泛化能力
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        self.dino_transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize image to mean and std of ImageNet

        head_feature_dim = feature_dim_dict[model_config['head']['model_type']]
        wrist_feature_dim = feature_dim_dict[model_config['wrist']['model_type']]
        feature_dim = max(head_feature_dim, wrist_feature_dim)
        # mask处理网络  ViT
        self.mask_process_net = nn.ModuleDict({
            # Convert 1-channel mask to patch embeddings
            'patch_embed': nn.Sequential(
                nn.Conv2d(1, head_feature_dim, kernel_size=14, stride=14),  # 分成14*14的非重叠patch
                nn.Flatten(2),  # Flatten H,W into patches
                Rearrange('b c n -> b n c'),
                nn.LayerNorm(head_feature_dim),
            ),
            # Process patch sequence with transformer blocks
            'transformer': nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=head_feature_dim,
                    nhead=8,
                    dim_feedforward=head_feature_dim*4,
                    dropout=0.0,
                    activation=nn.GELU(),
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=4
            )
        })

        # Create head_net to fuse rgb and mask features
        self.head_net = nn.Sequential(
            nn.Linear(head_feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.wrist_net = nn.Sequential(
            nn.Linear(wrist_feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # Create state_net to process robot arm and dexterous hand states
        self.state_net = nn.Sequential(
            nn.Linear(13, 256),   # TODO 13 is the state dimension？
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        self.shape_meta = shape_meta
        self.feature_dim = feature_dim

        logger.info(
            "Number of parameters in obs encoder: %e", sum(p.numel() for p in self.parameters())
        )

    def forward_head(self, rgbm_data, training=True):
        # rgbm_data: B,T,4,H,W

        B, T = rgbm_data.shape[:2]
        rgb_data = rgbm_data[:, :, :3] # [B, 1, 3, 518, 518]
        # Convert BGR to RGB since input images are in BGR format (from real robot camera)  # TODO 
        rgb_data = rgb_data[:, :, [2, 1, 0], ...]
        mask_data = rgbm_data[:, :, 3:]
        
        # Process RGB data
        rgb_data = rgb_data.reshape(B*T, *rgb_data.shape[2:]) # [8, 3, 518, 518]
        if training:
            # Only apply color jitter to RGB part
            rgb_data = self.color_jitter(rgb_data) # 色彩增强
        rgb_data = self.dino_transform(rgb_data) # 图片归一化
        with torch.no_grad():  # 禁用梯度计算 直接获取特征
            rgb_feature = self.dino_head.get_intermediate_layers(rgb_data, n=1)[0]  # [8, 1369, 768] patch=14是固定值 1369=（518/14）^2
            
        # Process mask data  ViT
        mask_data = mask_data.reshape(B*T, *mask_data.shape[2:]) # [8, 1, 518, 518]
        mask_feature = self.mask_process_net['patch_embed'](mask_data) # (B*T, num_patches, head_feature_dim)
        mask_feature = self.mask_process_net['transformer'](mask_feature) # [8, 1369, 768]
        
        # Fuse features
        combined_feature = torch.cat([rgb_feature, mask_feature], dim=-1) # [8, 1369, 1536]
        head_feature = self.head_net(combined_feature)  # (B*T, num_patches, feature_dim) mlp
        head_feature = head_feature.reshape(B, T*head_feature.shape[1], head_feature.shape[-1])  # (B, T*num_patches, feature_dim)[8, 1369, 1024])
        
        return head_feature
        
    def forward_wrist(self, wrist_data, training=True):
        # wrist_data: B,T,3,H,W
        B, T = wrist_data.shape[:2]
        # Convert BGR to RGB since input images are in BGR format (from real robot camera)
        wrist_data = wrist_data[:, :, [2, 1, 0], ...]
        wrist_data = wrist_data.reshape(B*T, *wrist_data.shape[2:])
        if training:
            wrist_data = self.color_jitter(wrist_data)
        wrist_data = self.dino_transform(wrist_data)
        with torch.no_grad():
            wrist_feature = self.dino_wrist.get_intermediate_layers(wrist_data, n=1)[0] # dino获取特征
        wrist_feature = self.wrist_net(wrist_feature) # MLP
        wrist_feature = wrist_feature.reshape(B, T*wrist_feature.shape[1], wrist_feature.shape[-1])  # (B, T*num_patches, feature_dim)
        return wrist_feature
        
    def forward_state(self, state_data):
        # state_data: B,T,13
        B, T = state_data.shape[:2]
        state_data = state_data.reshape(B*T, -1) # ([8, 13]
        state_feature = self.state_net(state_data)  # (B*T, feature_dim)
        state_feature = state_feature.reshape(B, T, state_feature.shape[-1])  # (B, T, feature_dim)
        return state_feature

    def forward(self, obs_dict, training=True):
        """
        Input:
        obs_dict = {
            'rgbm': (B,T,4,H,W),      # Head camera RGBM image
            'right_cam_img': (B,T,3,H,W), # Wrist camera RGB image  
            'right_state': (B,T,13)    # Robot arm state
        }
        Output:
        embeddings: (B,T*(num_patches*2+1),feature_dim) # Concatenate all features along sequence length dimension
                                                       # head and wrist each output T*num_patches features
                                                       # state outputs T features
        """
        embeddings = list()
        embeddings.append(self.forward_head(obs_dict['rgbm'], training))  # (B,T*num_patches,feature_dim)  B 1369 1024
        embeddings.append(self.forward_wrist(obs_dict['right_cam_img'], training))  # (B,T*num_patches,feature_dim)
        embeddings.append(self.forward_state(obs_dict['right_state']))  # (B,T,feature_dim)
        
        # Concatenate all features along sequence length dimension
        return torch.cat(embeddings, dim=1)  # (B,T*(num_patches*2+1),feature_dim) [8, 2739, 1024]  

    @torch.no_grad()
    def output_shape(self):
        return (1, 2739, self.feature_dim), [1369, 1369, 1]