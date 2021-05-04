from __future__ import absolute_import

#import epdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################################

class SiamNet_D(nn.Module):
    def __init__(self):
        super(SiamNet_D, self).__init__()
#        self.config = Config()
        self.nc = 3
        self.discriminator = nn.Sequential(
            
            OrderedDict([
	            # 3x127x127 - (3,64,3,2)
	            ('conv1', nn.Conv2d(self.nc, 64, 3, 2, bias=True) ),
	            ('bn1',	nn.BatchNorm2d(64) ),
	            ('lrelu1', nn.LeakyReLU(0.01,inplace=True) ),

                # 64x63x63 - (64,64,3,2)
                ('pool1', nn.MaxPool2d(3, 2)),
	            
	            # 64x31x31 - (64,128,3,2)
	            ('conv2', nn.Conv2d(64, 128, 3, 2, bias=True) ),
	            ('bn2',	nn.BatchNorm2d(128) ),
	            ('lrelu2', nn.LeakyReLU(0.01, inplace=True) ),

                # 128x15x15 - (128,128,3,2)
                ('pool2', nn.MaxPool2d(3, 2)),
	            
	            # 128x7x7 - (128,256,3,1)
	            ('conv3', nn.Conv2d(128, 256, 3, 1, bias=True) ),
	            ('bn3',	nn.BatchNorm2d(256) ),
	            ('lrelu3', nn.LeakyReLU(0.01,inplace=True) ),
	            
	            # 256x5x5 - (256,512,3,1)
	            ('conv4', nn.Conv2d(256, 512, 3, 1, bias=True) ),
	            ('bn4',	nn.BatchNorm2d(512) ),
	            ('lrelu4', nn.LeakyReLU(0.01,inplace=True) ),
	            
	            # 512x3x3 - (512,1,3,1)
	            ('conv5', nn.Conv2d(512, 1, 3, 1, bias=True) ),
	            ('sig1', nn.Sigmoid() )
            ])
        )
        
        # initialize weights
        self._initialize_weight() 
        
        
    def forward(self, inputs):
        return self.discriminator(inputs)
    
    def _initialize_weight(self):
        """initialize network parameters"""
        initD = 'truncated'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming initialization
#                if self.config.initD == 'kaiming':
#                    nn.init.kaiming_normal(m.weight.data, mode='fan_out')
#                    
#                # xavier initialization
#                elif self.config.initD == 'xavier':
#                   nn.init.xavier_normal(m.weight.data)
#                    m.bias.data.fill_(.1)
#
                if initD == 'truncated':
                    def truncated_norm_init(data, stddev=.01):
                        weight = np.random.normal(size=data.shape)
                        weight = np.clip(weight,
                                         a_min=-2*stddev, a_max=2*stddev)
                        weight = torch.from_numpy(weight).float()
                        return weight
                    m.weight.data = truncated_norm_init(m.weight.data)
                    #m.bias.data.fill_(.1)

                else:
                    raise NotImplementedError
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = 0.0003

#######################################################################################
class lossfn():
    def adversarial_loss(self, prediction, label, weight):
        #  weighted BCELoss 
        return F.binary_cross_entropy(prediction,
                                                  label,
                                                  weight)

    def mse_loss(self, prediction, label):
        return F.mse_loss(prediction, label)

    def hinge_loss(self, prediction, label):
        #  HingeEmbeddingLoss
        return F.hinge_embedding_loss(prediction,
                                     label,
                                     margin=1.0)

    def kldiv_loss(self, prediction, label):
        #  Kullback-Leibler divergence Loss.
        return F.kl_div(prediction,
                         label, 
                         reduction='batchmean')

    def weight_loss(self, prediction, label, weight):
        # weighted sigmoid cross entropy loss
        return F.binary_cross_entropy_with_logits(prediction.float(),label.float(),weight)

    def customize_loss(self, prediction, label, weight):
        score, y, weights = prediction, label, weight

        a = -(score * y)
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        loss = torch.mean(weights * loss)
        return loss


if __name__=='__main__':
    netD = SiamNet_D()
