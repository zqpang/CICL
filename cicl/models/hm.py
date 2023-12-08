import torch

from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('labels2', torch.zeros(num_samples).long())
        

    def forward(self, inputs, mask_inputs_full, indexes, epoch, back=1):
        # inputs: B*2048, features: L*2048
        targets = self.labels[indexes].clone()
        targets2 = self.labels2[indexes].clone()
        
        old_inputs = inputs.clone()
        inputs = hm(inputs, indexes, self.features, self.momentum)
        
        instances = inputs.clone() /0.09 
        
        inputs /= self.temp
        B = inputs.size(0)
        
        labels = self.labels.clone()
        labels2 = self.labels2.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        
        mask = mask.expand_as(sim)
        
        
        if back == 1:
            focal_loss = self.focal_loss(targets.detach().clone(), sim.clone(), mask.clone())
            contras_loss = self.contrasloss(old_inputs.detach().clone(), mask_inputs_full.clone())
            
            if epoch>=30:
                ins_loss = self.ins_loss(labels.detach().clone(), labels2.detach().clone(), targets.detach().clone(), targets2.detach().clone(), instances.clone())
                return focal_loss + contras_loss*0.25 + ins_loss*0.2
            else:
                return focal_loss + contras_loss*0.25
        
        
        elif back == 2:
            focal_loss = self.focal_loss(targets.detach().clone(), sim.clone(), mask.clone())
            contras_loss = self.contrasloss(old_inputs.detach().clone(), mask_inputs_full.clone())

            return focal_loss + contras_loss*0.25
        
        else:
            focal_loss = self.focal_loss(targets.detach().clone(), sim.clone(), mask.clone())
            return focal_loss
                    

    def focal_loss(self,targets ,sim, mask):
        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        targets_onehot = torch.zeros(masked_sim.size()).cuda()
        targets_squeeze = torch.unsqueeze(targets, 1)
        targets_onehot.scatter_(1, targets_squeeze, float(1))
        
        target_ones_p = targets_onehot.clone()
        focal_p  =target_ones_p.clone() * masked_sim.clone()
        focal_p_all = torch.pow(target_ones_p - focal_p, 2)

                
        outputs = torch.log(masked_sim+1e-6).float()
        loss = - (focal_p_all * outputs).float()
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)     
        return loss
    
    def contrasloss(self, inputs, another_inputs):
        inputs = (inputs.t() / inputs.norm(dim =1)).t()
        another_inputs = (another_inputs.t() / another_inputs.norm(dim =1)).t()
        loss = -1*(inputs * another_inputs).sum(dim = 1).mean()
        return loss


    def ins_loss(self, labels, labels2, targets, targets2, instances):
        
        loss = 0

        for i in range(len(instances)):
            indices = torch.nonzero(torch.eq(labels, targets[i]))
            indices2 = torch.nonzero(torch.eq(labels2, targets2[i]))
            pos_inds = torch.masked_select(indices, ~indices.unsqueeze(1).eq(indices2).any(1))
            neg_inds = torch.masked_select(indices2, ~indices2.unsqueeze(1).eq(indices).any(1))
            
            if (len(pos_inds) != 0 and len(neg_inds) != 0):
                pos_exps = torch.exp(instances[i][pos_inds]).sum()
                neg_exps = torch.exp(instances[i][neg_inds]).sum()
                ins = pos_exps/(pos_exps + neg_exps + 1e-6)
                loss += (- torch.log(ins+1e-6).float()/len(pos_inds)).float()
        loss /= len(instances)
        
        return loss