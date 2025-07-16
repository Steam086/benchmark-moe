import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized

import os



_EP: GroupCoordinator

_TP: GroupCoordinator

def init_env(backend: str, ep_size = 1, tp_size = 1) -> tuple[GroupCoordinator, GroupCoordinator]:

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    node_id = global_rank // local_world_size
    device_id = global_rank % local_world_size
    

    group_ranks = torch.arange(world_size).reshape(ep_size, tp_size)
    
    ep_ranks = group_ranks.transpose(0, 1).view(-1, ep_size).unbind(0)
    ep_ranks = [x.tolist() for x in ep_ranks]

    tp_ranks = group_ranks.view(-1, tp_size).unbind(0)
    tp_ranks = [x.tolist() for x in tp_ranks]
    

    _EP = GroupCoordinator(
        ep_ranks,
        global_rank,
        backend,
        True,
        group_name="ep"
    )

    _TP = GroupCoordinator(
        tp_ranks,
        global_rank,
        backend,
        True,
        group_name="tp"
    )
    return _EP, _TP


def get_ep_group():
    assert _EP is not None
    return _EP

def get_tp_group():
    assert _TP is not None
    return _TP