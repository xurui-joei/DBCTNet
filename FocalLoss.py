import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=3, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            #self.alpha = (torch.tensor(alpha)).view(-1,1).cuda()
            self.alpha = torch.tensor(alpha).view(-1,1).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0) #0.0001
        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        
        if self.use_alpha:
            alpha = self.alpha[target]
            #alpha = alpha.cuda()
            batch_loss = - alpha.double() * torch.pow(1-prob,self.gamma).double() * (prob.log()).double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        
        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss