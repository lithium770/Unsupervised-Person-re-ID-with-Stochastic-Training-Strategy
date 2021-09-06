import numpy as np

import torch
import torch.nn.functional as F
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
        #outputs = inputs.mm(ctx.instance_features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.cluster_features)
            #grad_inputs = grad_outputs.mm(ctx.instance_features)

        # momentum update
        for x, y, z in zip(inputs, indexes, targets):
            #if z > 0:
            ctx.cluster_features[z] = ctx.momentum * ctx.cluster_features[z] + (1. - ctx.momentum) * x
            ctx.cluster_features[z] /= ctx.cluster_features[z].norm()

            ctx.instance_features[y] = ctx.momentum * ctx.instance_features[y] + (1. - ctx.momentum) * x
            ctx.instance_features[y] /= ctx.instance_features[y].norm()

        return grad_inputs, None, None, None, None, None, None


def hm(inputs, indexes, targets, instance_features, cluster_features, momentum=0.5):
    return HM.apply(inputs, indexes, targets, instance_features, cluster_features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, num_source=0):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.camera_features = []
        self.concate_intra_class = []

        # cluster memory: store features for cluster centroids
        self.register_buffer('cluster_features', torch.zeros(num_samples, num_features))
        # instance memory: store features for instances
        self.register_buffer('instance_features', torch.zeros(num_samples, num_features))
        # pseudo labels without outliers
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes, cids=None, trans=None):
        # inputs: B*2048, features: L*2048
        B = inputs.size(0)

        labels = self.labels.clone()
        targets = labels[indexes].clone()

        inputs = hm(inputs, self.global_ind[indexes], targets, self.instance_features, self.cluster_features,
                    self.momentum)

        inputs /= self.temp

        def softmax(vec, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_sums = exps.clone().sum(dim, keepdim=True) + epsilon
            return (exps / masked_sums)

        masked_sim = softmax(inputs)

        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
