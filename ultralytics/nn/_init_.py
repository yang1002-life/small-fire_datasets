# Ultralytics YOLO11 🚀
"""
Ultralytics modules.

Example:
    Visualize a module.
"""
# ------------------------------------------------------------------------------------------------------------------------
from ultralytics.nn.core11.emo import C3_RMB, CSRMBC, C2f_RMB, CPNRMB, ReNLANRMB
from ultralytics.nn.core11.biformer import CSCBiF, ReNLANBiF, CPNBiF, C3_Biformer, C2f_Biformer
from ultralytics.nn.core11.CFNet import CSCFocalNeXt, ReNLANFocalNeXt, CPNFocalNeXt, C3_FocalNeXt, C2f_FocalNeXt
from ultralytics.nn.core11.FasterNet import (
    FasterNeXt, CSCFasterNeXt, ReNLANFasterNeXt, C3_FasterNeXt, C2f_FasterNeXt
)
from ultralytics.nn.core11.Ghost import CPNGhost, CSCGhost, ReNLANGhost, C3_Ghost, C2f_Ghost

from ultralytics.nn.core11.EfficientRep import RepVGGBlock, SimConv, RepBlock, Transpose
from ultralytics.nn.core11.damo import CReToNeXt
from ultralytics.nn.core11.MobileViTv1 import CPNMobileViTB, CSCMobileViTB, ReNLANMobileViTB, C3_MobileViTB, C2f_MobileViTB

from ultralytics.nn.core11.QARep import QARep, CSCQARep, ReNLANQARep, C3_QARep, C2f_QARep, QARepNeXt
from ultralytics.nn.core11.ConvNeXtv2 import CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2
from ultralytics.nn.core11.MobileViTv2 import CPNMVBv2, CSCMVBv2, ReNLANMVBv2, C3_MVBv2, C2f_MVBv2
from ultralytics.nn.core11.MobileViTv3 import CPNMViTBv3, CSCMViTBv3, ReNLANMViTBv3, C3_MViTBv3, C2f_MViTBv3
from ultralytics.nn.core11.RepLKNet import CPNRepLKBlock, CSCRepLKBlock, ReNLANRepLKBlock, C3_RepLKBlock, C2f_RepLKBlock
from ultralytics.nn.core11.DenseNet import CPNDenseB, CSCDenseB, ReNLANDenseB, C3_DenseB, C2f_DenseB
from ultralytics.nn.core11.MSBlock import CPNMSB, CSCMSB, ReNLANMSB, C3_MSB, C2f_MSB
from ultralytics.nn.core11.GhostNetv2 import CPNGhostblockv2, CSCGhostblockv2, ReNLANGhostblockv2, C3_Ghostblockv2, C2f_Ghostblockv2

from ultralytics.nn.core11.shiftconv import ReparamLKBlock
from ultralytics.nn.core11.efficientnet import stems, MBConvBlock
from ultralytics.nn.core11.efficientnetv2 import FusedMBConv, MBConv
from ultralytics.nn.core11.swin import CSwinTR, SwinTRX, SwinTRY, SwinTRZ, SwinV2TRX, SwinV2TRY, SwinV2TRZ, CSwinTRv2
from ultralytics.nn.core11.ema import EMA

from ultralytics.nn.core11.CARAFE import CARAFE
from ultralytics.nn.core11.GELAN import SPPELAN, RepNCSPELAN4, ADown
from ultralytics.nn.core11.AFPN import ASFF_3, ASFF_2, BasicBlock
from ultralytics.nn.core11.SSFF import SSFF

from ultralytics.nn.core11.GDM import (LAF_px, low_FAM, LAF_h, low_IFM, InjectionMultiSum_Auto_pool1, 
InjectionMultiSum_Auto_pool2, InjectionMultiSum_Auto_pool3, InjectionMultiSum_Auto_pool4, 
PyramidPoolAgg, TopBasicLayer)

from ultralytics.nn.core11.RepFPN import SimSPPF
from ultralytics.nn.core11.ASPP import ASPP

from ultralytics.nn.core11.BasicRFB import BasicRFB
from ultralytics.nn.core11.SPPFCSPC import SPPFCSPC
from ultralytics.nn.core11.SimSPPF import BiFusion



from ultralytics.nn.core11.resnet import ResNet50vd,ResNet50vd_dcn,ResNet101vd,PPConvBlock,CoordConv,Res2net50

from ultralytics.nn.core11.Dysample import DySample

# from ultralytics.nn.core11.arconv import ARConv, CARC

from ultralytics.nn.core11.MAF import Stem,ConvWrapper,AVG_down, CSPDepthResELAN, MPRep, SimConvWrapper, CSPRepResELAN, RepELANMSv2, CSPSDepthResELAN, SDepthMP, RepHDW, RepELANMS,RepELANMS2

# from ultralytics.nn.core11.Attention.mlla import LinearAttention

from ultralytics.nn.core11.SPPELAN import SPPELAN

from ultralytics.nn.core11.Attention.simam import SimAM
from ultralytics.nn.core11.Attention.gam import GAMAttention
from ultralytics.nn.core11.Attention.cbam import CBAM
from ultralytics.nn.core11.Attention.sk import SKAttention
from ultralytics.nn.core11.Attention.soca import SOCA
from ultralytics.nn.core11.Attention.sa import ShuffleAttention