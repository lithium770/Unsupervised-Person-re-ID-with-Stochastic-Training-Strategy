import numpy as np

import torch
import torch.nn.functional as F
import random
from torch.nn import init
from torch import nn, autograd


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, targets, instance_features, cluster_features, momentum):
        ctx.instance_features = instance_features
        ctx.cluster_features = cluster_features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes, targets)
        outputs = inputs.mm(ctx.cluster_features.t())
        # outputs = inputs.mm(ctx.instance_features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.cluster_features)
            # grad_inputs = grad_outputs.mm(ctx.instance_features)

        # momentum update
        for x, y, z in zip(inputs, indexes, targets):
            ctx.cluster_features[z] = ctx.momentum * ctx.cluster_features[z] + (1-ctx.momentum) * x
            ctx.cluster_features[z] /= ctx.cluster_features[z].norm()
            #ctx.cluster_features[y] = x
            #ctx.cluster_features[y] /= ctx.cluster_features[y].norm()

            ctx.instance_features[y] = ctx.momentum * ctx.instance_features[y] + (1-ctx.momentum) * x
            ctx.instance_features[y] /= ctx.instance_features[y].norm()

        return grad_inputs, None, None, None, None, None, None


def hm(inputs, indexes, targets, instance_features, cluster_features, momentum=0.5):
    return HM.apply(inputs, indexes, targets, instance_features, cluster_features,
                    torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, num_source=0):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.num_source = num_source

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('cluster_features', torch.zeros(num_samples, num_features))
        self.register_buffer('instance_features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam_labels', torch.zeros(num_samples).int())

    def forward(self, inputs, indexes, is_target=False):
        # inputs: B*2048, features: L*2048
        B = inputs.size(0)

        labels = self.labels.clone()
        targets = labels[indexes].clone()

        if is_target:
            indexes = self.global_ind[indexes-self.num_source]

        inputs = hm(inputs, indexes, targets, self.instance_features, self.cluster_features, self.momentum)

        inputs /= self.temp

        def softmax(vec, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            # sorted, _ = torch.sort(masked_exps, descending=True)
            # masked_sums = torch.zeros([B, 1]).cuda()
            # for i in range(B):
            #    masked_sums[i] = sorted[i][:50].unsqueeze(0).sum(dim, keepdim=True) + epsilon
            masked_sums = exps.clone().sum(dim, keepdim=True) + epsilon
            return (exps / masked_sums)

        masked_sim = softmax(inputs)

        #sim = torch.zeros(labels.max() + 1, B).float().cuda()
        """
        for c in range(labels.max() + 1):
            pic_num = int(len(self.tar_ind[c]) * self.mean_factor)
            if pic_num < 1:
                pic_num = 1
            ex_ind = self.tar_ind[c] + self.tar_ind[c]
            sim[c] = torch.sum(inputs[:, ex_ind[int(self.current_ind[c]):int(self.current_ind[c])+pic_num]], dim=1) / pic_num
            self.current_ind[c] = (self.current_ind[c] + pic_num) % len(self.tar_ind[c])

        sim.index_add_(0, labels, torch.index_select(inputs, 1, self.global_ind).t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(labels.size(0), 1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        """
        """
        # choose random instance to momentum represent cluster centroids
        for c in range(labels.max()+1):
            pic_ind = self.tar_ind[c][self.current_ind[c]]
            sim[c] = inputs[:, pic_ind].t()
            self.current_ind[c] = (self.current_ind[c] + 1) % len(self.tar_ind[c])

        masked_sim = softmax(sim.t().contiguous())
        """

        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
