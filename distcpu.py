import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from torch.nn.modules import Module
from collections import defaultdict
from torch.autograd import Variable


class DistributedDataParallelCPU(Module):

    def __init__(self, module):
        super(DistributedDataParallelCPU, self).__init__()
        self.module = module
        self.init_sync_parameters()
        self.needs_reduction = True

        # def allreduce_params():
        #     if self.needs_reduction:
        #         print("Reducing", dist.get_rank())
        #         self.needs_reduction = False
        #         buckets = defaultdict(list)
        #         for param in self.module.parameters():
        #             if param.requires_grad and param.grad is not None:
        #                 tp = type(param.data)
        #                 buckets[tp].append(param)

        #         for bucket in buckets.values():
        #             grads = [param.grad.data for param in bucket]
        #             coalesced = _flatten_dense_tensors(grads)
        #             dist.all_reduce(coalesced)
        #             coalesced /= dist.get_world_size()
        #             for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        #                 buf.copy_(synced)

        # for param in list(self.module.parameters()):
        #     def allreduce_hook(*unused):
        #         Variable._execution_engine.queue_callback(allreduce_params)

        #     if param.requires_grad:
        #         param.register_hook(allreduce_hook)

    def init_sync_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, 0)

    def sync_parameters(self):
        # print("%%%% Sync params %%%%")
        buckets = defaultdict(list)
        for param in self.module.parameters():
            tp = type(param.data)
            buckets[tp].append(param)

        for bucket in buckets.values():
            ps = [param.data for param in bucket]
            coalesced = _flatten_dense_tensors(ps)
            dist.all_reduce(coalesced)
            coalesced /= dist.get_world_size()
            for buf, syncd in zip(ps, _unflatten_dense_tensors(coalesced, ps)):
                buf.copy_(syncd)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
