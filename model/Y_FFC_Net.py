import torch
import torch.nn as nn
from torchvision import models
from ffc import *

class Conv2D_BN_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding=0,
            dilation=1, if_bn=True, activation='relu', bias=None, initializer=None, transpose=False
    ):
        super(Conv2D_BN_activa, self).__init__()
        if transpose:
            self.conv2d = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, bias=(not if_bn) if bias is None else bias
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, bias=(not if_bn) if bias is None else bias
            )
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)  # eps same as that in the official codes.
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
        self.initializer = initializer
        if self.initializer is not None:
            if self.initializer == 'truncated_norm':
                nn.init.normal_(self.conv2d.weight, std=0.02)
                self.conv2d.weight = truncated_normal_(self.conv2d.weight, std=0.02)

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation:
            if self.if_bn:
                x = self.bn(x)
            x = self.activation(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class FeatureExtrationLayersA(nn.Module):
    def __init__(self, in_channels, out_channels_branch=192):
        super(FeatureExtrationLayersA, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 96, 1, 1, 0),
            Conv2D_BN_activa(96, out_channels_branch, 7, 1, 3)
        )
        self.path2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 256, 1, 1, 0),
            Conv2D_BN_activa(256, 256, 3, 1, 1),
            Conv2D_BN_activa(256, out_channels_branch, 3, 1, 1)
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x1, x2, x3), 1)

        return out


class FeatureExtrationLayersB(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtrationLayersB, self).__init__()
        self.path1 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 7, 1, 3)
        )
        self.path2 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 192, 3, 1, 1)
        )
        self.path3 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1)
        )
        self.path4 = nn.Sequential(
            Conv2D_BN_activa(in_channels, 128, 1, 1, 0),
            Conv2D_BN_activa(128, 128, 3, 1, 1),
            Conv2D_BN_activa(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)

        out = torch.cat((path1, path2, path3, path4), 1)
        return out


class ImageDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels_branch=256):
        super(ImageDecBlock, self).__init__()
        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 4, 2, 1, transpose=True)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_1(x)
        x_3 = self.branch_1(x)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class ImageDecBlock_same(nn.Module):
    def __init__(self, in_channels, out_channels_branch=256):
        super(ImageDecBlock_same, self).__init__()

        self.branch_1 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_2 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)
        self.branch_3 = Conv2D_BN_activa(in_channels, out_channels_branch, 3, 1, 1)

    def forward(self, x):
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_123 = torch.cat([x_1, x_2, x_3], dim=1)
        return x_123

class CencN(nn.Module):
    def __init__(self):
        super(CencN, self).__init__()
        backbone_model = models.vgg16_bn(pretrained=True)
        backbone_model_list = list(backbone_model.features.children())
        self.backbone_1 = nn.Sequential(*backbone_model_list[0:7])
        self.backbone_2 = nn.Sequential(*backbone_model_list[7:14])
        self.backbone_3 = nn.Sequential(*backbone_model_list[14:24])
        self.backbone_4 = nn.Sequential(*backbone_model_list[24:34])
        self.backbone_5 = nn.Sequential(*backbone_model_list[34:44])
        self.conv1 = Conv2D_BN_activa(1, 3, 3, 1, 1)
        self.cba_after_backbone = Conv2D_BN_activa(512, 256, 3, 1, 1)

    def forward(self, x):
        x1=self.conv1(x)
        x_bb_1 = self.backbone_1(x1)
        x_bb_2 = self.backbone_2(x_bb_1)
        x_bb_3 = self.backbone_3(x_bb_2)
        x_bb_4 = self.backbone_4(x_bb_3)
        x_c = self.cba_after_backbone(x_bb_4)


        return x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c


class GdecN(nn.Module):
    def __init__(self):
        super(GdecN, self).__init__()
        self.block_1_conv_1 = Conv2D_BN_activa(512, 1024, 7, 1, 3)
        self.block_1_conv_2 = Conv2D_BN_activa(1024, 512, 1, 1, 0)
        self.block_1_conv_3 = Conv2D_BN_activa(512, 256, 3, 1, 1)
        self.block_1_conv_transpose = Conv2D_BN_activa(256, 256, 5, 1, 2, transpose=False)

        self.block_2_conv = Conv2D_BN_activa(256 + 512, 128, 3, 1, 1)
        self.block_2_conv_transpose = Conv2D_BN_activa(128, 128, 4, 2, 1, transpose=True)

        self.block_3_conv = Conv2D_BN_activa(128 + 256, 64, 3, 1, 1)
        self.block_3_conv_transpose = Conv2D_BN_activa(64, 64, 4, 2, 1, transpose=True)

        self.block_4_conv = Conv2D_BN_activa(64 + 128, 32, 3, 1, 1)
        self.block_4_conv_transpose = Conv2D_BN_activa(32, 32, 4, 2, 1, transpose=True)

        self.block_5_conv_1 = Conv2D_BN_activa(32 + 64, 64, 4, 2, 1, transpose=True)
        self.block_5_conv_2 = Conv2D_BN_activa(64, 1, 5, 1, 2, activation=None, if_bn=False)

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, backbone_features):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4 = backbone_features
        x = self.block_1_conv_1(x_bb_4)
        x = self.block_1_conv_2(x)
        x = self.block_1_conv_3(x)
        x_to_be_enhanced_1 = self.block_1_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_1, x_bb_4], dim=1)

        x = self.block_2_conv(x)
        x_to_be_enhanced_2 = self.block_2_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_2, x_bb_3], dim=1)

        x = self.block_3_conv(x)
        x_to_be_enhanced_3 = self.block_3_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_3, x_bb_2], dim=1)

        x = self.block_4_conv(x)
        x_to_be_enhanced_4 = self.block_4_conv_transpose(x)
        x = torch.cat([x_to_be_enhanced_4, x_bb_1], dim=1)

        x = self.block_5_conv_1(x)
        x = self.block_5_conv_2(x)

        x_g = self.sigmoid_layer(x)

        return x_to_be_enhanced_1, x_to_be_enhanced_2, x_to_be_enhanced_3, x_to_be_enhanced_4, x_g


class IdecN(nn.Module):
    def __init__(self, in_channels):
        super(IdecN, self).__init__()
        self.feature_extraction_layers_A = FeatureExtrationLayersA(in_channels)
        self.image_dec_block_1 = ImageDecBlock_same(192 * 3, 256)
        self.image_dec_block_2 = ImageDecBlock(256 * 3 + 128 + 512, 128)  # channels of [image_dec_block,
        self.feature_extraction_layers_B = FeatureExtrationLayersB(
            128 * 3 + 64 + 256)  # feature_enhancement_layers, vgg16]
        self.image_dec_block_3 = ImageDecBlock(192 + 192 + 128 + 128, 64)
        self.image_dec_block_4 = ImageDecBlock(64 * 3 + 32 + 128, 32)
        self.image_dec_block_5 = ImageDecBlock(32 * 3 + 16 + 64, 16)
        self.cba_output_1 = Conv2D_BN_activa(16 * 3 + 1, 16, 3, 1, 1)
        self.cba_output_2 = Conv2D_BN_activa(16, 1, 3, 1, 1)
        self.FFC1 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5)
        self.FFC2 = FFC_BN_ACT(256, 256, 3, 0.5, 0.5)
        self.att1=Attention_block(768,512,640)
        self.att2 = Attention_block(384, 256, 320)
        self.att3 = Attention_block(192, 128, 160)
        self.att4 = Attention_block(96, 64, 80)

    def forward(self, c_features, enhanced_features, x_g):
        x_bb_1, x_bb_2, x_bb_3, x_bb_4, x_c = c_features
        x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4 = enhanced_features
        x0, x1 = x_c.chunk(2, dim=1)
        x=self.FFC1(tuple([x0,x1]))
        x=self.FFC2(x)
        x = torch.cat([x[0], x[1]], dim=1)
        x=x+x_c
        x = self.feature_extraction_layers_A(x)
        x = self.image_dec_block_1(x)
        bridge=self.att1(x,x_bb_4)
        x = torch.cat([x, bridge,x_enhanced_1], dim=1)

        x = self.image_dec_block_2(x)
        bridge = self.att2(x, x_bb_3)
        x = torch.cat([x, bridge, x_enhanced_2], dim=1)

        x = self.feature_extraction_layers_B(x)
        x = self.image_dec_block_3(x)
        bridge = self.att3(x, x_bb_2)
        x = torch.cat([x, bridge, x_enhanced_3], dim=1)

        x = self.image_dec_block_4(x)
        bridge=self.att4(x, x_bb_1)
        x = torch.cat([x, bridge, x_enhanced_4], dim=1)

        x = self.image_dec_block_5(x)
        x = torch.cat([x, x_g], dim=1)

        x = self.cba_output_1(x)
        est_b = self.cba_output_2(x)

        return est_b


class EnhancementLayers(nn.Module):
    def __init__(self):
        super(EnhancementLayers, self).__init__()
        self.enhancement_layers_1 = Conv2D_BN_activa(256, 128, 7, 1, 3)
        self.enhancement_layers_2 = Conv2D_BN_activa(128, 64, 7, 1, 3)
        self.enhancement_layers_3 = Conv2D_BN_activa(64, 32, 7, 1, 3)
        self.enhancement_layers_4 = Conv2D_BN_activa(32, 16, 7, 1, 3)

    def forward(self, to_be_enhanced_features):
        x_enhanced_1 = self.enhancement_layers_1(to_be_enhanced_features[0])
        x_enhanced_2 = self.enhancement_layers_2(to_be_enhanced_features[1])
        x_enhanced_3 = self.enhancement_layers_3(to_be_enhanced_features[2])
        x_enhanced_4 = self.enhancement_layers_4(to_be_enhanced_features[3])

        return x_enhanced_1, x_enhanced_2, x_enhanced_3, x_enhanced_4


class YFFC(nn.Module):
    def __init__(self):
        super(YFFC, self).__init__()
        self.encoder_context = CencN()
        self.decoder_gradient = GdecN()
        self.enhancement_layers = EnhancementLayers()
        self.decoder_image = IdecN(in_channels=256)


    def forward(self, x):
        c_features = self.encoder_context(x)
        g_features = self.decoder_gradient(c_features[:-1])
        enhanced_features = self.enhancement_layers(g_features[:-1])
        est_b = self.decoder_image(c_features, enhanced_features, g_features[-1])
        estimations = {'r': x - est_b, 'g': g_features[-1], 'b': est_b}
        return estimations

maxim = YFFC()
input = torch.zeros(size=(4, 1, 512, 512))
out = maxim(input)
print(out['g'].shape)