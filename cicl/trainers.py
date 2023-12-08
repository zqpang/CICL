from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter


class Mask_CACLSIC_USL(object):
    #def __init__(self, encoder1, encoder2, memory1, memory2):
    def __init__(self, encoder, encoder_fusion, memory1, memory2, memory3, memory_fusion):
    # def __init__(self, *encoders, memories):
        super(Mask_CACLSIC_USL, self).__init__()
        self.encoder = encoder
        #self.encoder2 = encoder2
        self.encoder_fusion = encoder_fusion
        self.memory1 = memory1
        self.memory2 = memory2
        self.memory3 = memory3
        self.memory4 = memory_fusion


    def train(self, epoch, data_loader1, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        #self.encoder2.train()
        #self.encoder4.train()
        self.encoder_fusion.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
        losses4 = AverageMeter()
        #losses5 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            
            data_time.update(time.time() - end)

            inputs1, inputs2, inputs3, _, indexes1 = self._parse_data(inputs1)
            
            #print("inputs1:",inputs1.size())
            #print("inputs2:",inputs2.size())

            bn_x1, full_conect1, bn_x2, full_conect2, bn_x3, full_conect3, fusion = self._forward(inputs1, inputs2, inputs3)

            #flag = 2
            loss1 = self.memory1(bn_x1, full_conect2.clone(), indexes1, epoch, back = 1)
            #flag = 1
            loss2 = self.memory2(bn_x2, full_conect1.clone(), indexes1, epoch, back = 2)
            #loss2 = self.memory2(bn_x2, bn_x1.clone(), indexes1, back = 2)
            #flag = 0
            #loss4 = self.memory4(fusion1, full_conect1.clone(), indexes1, back = 0)
            loss3 = self.memory3(bn_x3, full_conect1.clone(), indexes1, epoch, back = 3)
            loss4 = self.memory4(fusion, full_conect1.clone(), indexes1, epoch, back = 3)
            
            #loss = (loss1 + loss2 + loss4)
            loss = (loss1 + loss2 + loss3 + loss4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses1.update(loss1.item())
            losses2.update(loss2.item())
            losses3.update(loss3.item())
            losses4.update(loss4.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.5f} ({:.5f})\t'
                      'Data {:.5f} ({:.5f})\t'
                      'Loss_ALL {:.5f} ({:.5f})\t'
                      'Loss_RGB {:.5f} ({:.5f})\t'
                      'Loss_Mask {:.5f} ({:.5f})\t'
                      'Loss_black {:.5f} ({:.5f})\t'
                      'Loss_fusion {:.5f} ({:.5f})'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses1.val, losses1.avg,
                              losses2.val, losses2.avg,
                              losses3.val, losses3.avg,
                              losses4.val, losses4.avg
                              ))
                
                
    def _parse_data(self, inputs):
        img, imag_mask, imag_black, _, pids, clothesid, _, indexes = inputs
        return img.cuda(), imag_mask.cuda(), imag_black.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs1, inputs2, inputs3):
        
        bn_x1, full_conect1 = self.encoder(inputs1) #bn_x1.shape: torch.Size([64, 2048])
        bn_x2, full_conect2 = self.encoder(inputs2)
        bn_x3, full_conect3 = self.encoder(inputs3)
        #features_stack = torch.stack((bn_x1, bn_x3), dim=1)
        fusion = self.encoder_fusion(bn_x1, bn_x3)
        
        return bn_x1, full_conect1, bn_x2, full_conect2, bn_x3, full_conect3, fusion