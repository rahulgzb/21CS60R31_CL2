import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)

def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:,:,:, crop_t:-crop_b, crop_l:-crop_r]
    return x



####
class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride
    
    def forward(self, x):
        #print("sampling")
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start,pad_val_end, pad_val_start, pad_val_end)
        #print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x


####

####
class UpSample2xx(nn.Module):
    """Upsample input by a factor of 2.
    
    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2xx, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        #print(input_shape,"inside upsample")
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0,1,2,3,5,4,6)
        #print(ret.size())
        ret = ret.reshape((-1, input_shape[1],input_shape[2], input_shape[3] * 2, input_shape[4] * 2))
        return ret



class gendecoder(nn.Module):

    # input encoders for the model
    def __init__(self) :
        super(gendecoder,self).__init__()

        self.relu= nn.ReLU(inplace=True)
        self.upsample2xx=UpSample2xx()
    

        #bottleneck 
        self.neck1= P4MConvP4M(in_channels = 256, out_channels = 512 , kernel_size =3, stride = 1, padding = 0)
        self.neck2= P4MConvP4M(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 0)
        self.bnneck=nn.BatchNorm3d(512)
        self.u0= P4MConvP4M(in_channels = 512, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)


        #decoders 
        self.decode001= P4MConvP4M(in_channels = 256, out_channels = 256 , kernel_size = 3, stride = 1, padding = 0)
        self.decode010= P4MConvP4M(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0)
        self.ndn01=nn.BatchNorm3d(256)
        self.u1= P4MConvP4M(in_channels = 256, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)


        self.decode01= P4MConvP4M(in_channels = 128, out_channels = 128 , kernel_size = 3, stride = 1, padding = 0)
        self.decode10= P4MConvP4M(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.ndn1=nn.BatchNorm3d(128)
        self.u2 = P4MConvP4M(in_channels = 128, out_channels = 64, kernel_size = 1, stride = 1, padding = 0)

        self.decode02= P4MConvP4M(in_channels = 64, out_channels =64 , kernel_size = 5, stride = 1, padding = 0)
        self.decode20= P4MConvP4M(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        self.ndn2=nn.BatchNorm3d(64)
        self.u3 = P4MConvP4M(in_channels = 64, out_channels = 32, kernel_size = 1, stride = 1, padding = 0)

        self.decode03= P4MConvP4M(in_channels = 32, out_channels = 32 , kernel_size = 5, stride = 1, padding = 0)
        self.decode30= P4MConvP4M(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
        self.ndn3=nn.BatchNorm3d(32)
        self.u4 = P4MConvP4M(in_channels = 32, out_channels = 16, kernel_size = 1, stride = 1, padding = 0)


        #final layer
        self.decode04= P4MConvP4M(in_channels = 16, out_channels = 16 , kernel_size = 5, stride = 1, padding = 0)
        self.decode40= P4MConvP4M(in_channels = 16, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
        self.ndn4=nn.BatchNorm3d(16)
        self.ndn5=nn.BatchNorm3d(8)
        self.u5 = P4MConvP4M(in_channels = 16, out_channels = 8, kernel_size = 1, stride = 1, padding = 0)
        self.u6 = nn.Sequential( nn.Conv2d(kernel_size=1, in_channels=64, out_channels=2,stride=1, padding=0, bias=False),
                                #  nn.BatchNorm2d(64, eps=1e-5),
                                #  nn.ReLU(inplace=True),
                                #  nn.Conv2d(kernel_size=1, in_channels=64, out_channels=32,stride=1, padding=0, bias=False),
                                #  nn.BatchNorm2d(32, eps=1e-5),
                                #  nn.ReLU(inplace=True),
                                #  nn.Conv2d(kernel_size=1, in_channels=32, out_channels=2,stride=1, padding=0, bias=False),
                                 )
       

        

    def forward(self,x,e1,e2,e3,e4,e5):

        #bottle neck part 
        x=self.relu(self.bnneck(self.neck1(x)))
        x=self.relu(self.bnneck(self.neck2(x)))
        x=self.upsample2xx(x)
         # padding values for concatination
        x=self.relu(self.ndn01(self.u0(x)))
         # concatination of tensor values 
        #print(x.shape,e5.shape)
        x=x+e5
        
        
       
        
    
        #decoder part
        x=self.relu(self.ndn01(self.decode001(x)))
        x=self.relu(self.ndn01(self.decode010(x)))
        x=self.upsample2xx(x)
        x=self.relu(self.ndn1(self.u1(x)))
            #padding is required
        #print(x.shape,e4.shape)
        x=x+e4




        x=self.relu(self.ndn1(self.decode01(x)))
        x=self.relu(self.ndn1(self.decode10(x)))
        x=self.upsample2xx(x)
        x=self.relu(self.ndn2(self.u2(x)))
        #print(x.shape,e3.shape)
        x=x+e3 
        

        
        x=self.relu(self.ndn2(self.decode02(x)))
        x=self.relu(self.ndn2(self.decode20(x)))
        x=self.upsample2xx(x)
        x=self.relu(self.ndn3(self.u3(x)))
        #print(x.shape,e2.shape)
        x=x+e2
        

        
        x=self.relu(self.ndn3(self.decode03(x)))
        x=self.relu(self.ndn3(self.decode30(x)))
        x=self.relu(self.ndn4(self.u4(x)))
        #print(x.shape,e1.shape)
        x=x+e1


        

        #final layer 
        x=self.relu(self.ndn4(self.decode04(x)))
        x=self.relu(self.ndn4(self.decode40(x)))
        x=self.relu(self.ndn5(self.u5(x)))
        outs=x.size()
        x=x.view(outs[0],outs[1]*outs[2],outs[3],outs[4])
        x=self.relu(self.u6(x))
         
        return x
    







####
class HoVerNet(nn.Module):
    """Initialise HoVer-Net."""
          
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super(HoVerNet,self).__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4
        
        

        assert mode == 'original', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode

        # encoder  part 
        self.conv0 = P4MConvZ2(in_channels = 3, out_channels = 8, kernel_size = 7, stride = 1, padding = 0, bias=False)
        self.bn0 = nn.BatchNorm3d(8)
        self.relu= nn.ReLU(inplace=True)


        self.enconv01 =  P4MConvP4M(in_channels = 8, out_channels = 16, kernel_size = 7, stride = 1, padding = 0)
        self.enconv10 =  P4MConvP4M(in_channels = 16, out_channels = 16, kernel_size = 7, stride = 1, padding = 0)
        self.ebn1 =      nn.BatchNorm3d(16)
        self.maxpool1=    P4MConvP4M(in_channels = 16, out_channels = 16, kernel_size = 7, stride = 1, padding = 0)

    
        self.enconv02 =  P4MConvP4M(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
        self.enconv20 =  P4MConvP4M(in_channels = 32, out_channels =32, kernel_size = 5, stride = 1, padding = 0)
        self.ebn2 = nn.BatchNorm3d(32)
        self.maxpool2=    P4MConvP4M(in_channels = 32, out_channels = 32, kernel_size = 1, stride = 2, padding = 0)
         
        self.enconv03 =  P4MConvP4M(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        self.enconv30 =  P4MConvP4M(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        self.ebn3 = nn.BatchNorm3d(64)
        self.maxpool3=    P4MConvP4M(in_channels = 64, out_channels = 64, kernel_size = 1, stride = 2, padding = 0)
    
        self.enconv04 =  P4MConvP4M(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.enconv40 =  P4MConvP4M(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.ebn4 = nn.BatchNorm3d(128)
        self.maxpool4=    P4MConvP4M(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 2, padding = 0)

        self.enconv05 =  P4MConvP4M(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 0)
        self.enconv50 =  P4MConvP4M(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0)
        self.ebn5 = nn.BatchNorm3d(256)
        self.maxpool5=    P4MConvP4M(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 2, padding = 0)

        # TODO: pytorch still require the channel eventhough its ignored
        self.gendecoders= gendecoder()
    

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY
        # TODO: switch to `crop_to_shape` ?
        x=self.relu(self.bn0(self.conv0(imgs)))
        #encoder
        x=self.relu(self.ebn1(self.enconv01(x)))
        #x=self.relu(self.ebn1(self.enconv10(x)))
        xe1=x
        x=self.relu(self.ebn1((self.maxpool1(x))))
    
       
        x=self.relu(self.ebn2(self.enconv02(x)))
        #x=self.relu(self.ebn2(self.enconv20(x)))
        xe2=x
        x=self.relu(self.ebn2((self.maxpool2(x))))
        
        
        x=self.relu(self.ebn3(self.enconv03(x)))
        #x=self.relu(self.ebn3(self.enconv30(x)))
        xe3=x
        x=self.relu(self.ebn3((self.maxpool3(x))))
        

        x=self.relu(self.ebn4(self.enconv04(x)))
        #x=self.relu(self.ebn4(self.enconv40(x)))
        xe4=x
        x=self.relu(self.ebn4((self.maxpool4(x))))
      
        
        x=self.relu(self.ebn5(self.enconv05(x)))
        #x=self.relu(self.ebn5(self.enconv50(x)))
        xe5=x
        x=self.relu(self.ebn5((self.maxpool5(x))))

        #croping of from center 
        if self.mode == 'original':
           xe1 = crop_op(xe1, [170, 170])
           xe2 = crop_op(xe2, [152, 152])
           xe3 = crop_op(xe3, [64, 64])
           xe4 = crop_op(xe4, [26, 26])
           xe5 = crop_op(xe5, [7, 7])

        #decoder part here 
        x1=x
        x=self.gendecoders(x1,xe1,xe2,xe3,xe4,xe5)
        y=self.gendecoders(x1,xe1,xe2,xe3,xe4,xe5)

        #decode 
        out_dict = OrderedDict()
        out_dict={'np':x ,'hv':y}
        return out_dict


####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return HoVerNet(mode=mode, **kwargs)



def test():
    net = create_model('original')
    #print(net)
    y = net(Variable(torch.randn(2,3,270,270)))
    print(y['hv'].size())
    print("hi")

#test()