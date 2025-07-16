
import torch
from typing import Any, Callable, List, Optional, Tuple, Union
from vllm_ascend.ops.fused_moe import select_experts, fused_experts_with_all2all
from vllm.distributed.parallel_state import GroupCoordinator



def fused_moe(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    ep_group: GroupCoordinator,
    renormalize: bool,
    use_grouped_topk: bool,
    expert_map: Optional[torch.Tensor] = None,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    is_prefill: bool = False,
    **kwargs,
):
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
    )

    topk_weights = topk_weights.to(x.dtype)

    return fused_experts_with_all2all(hidden_states=x,
                                      w1=w1,
                                      w2=w2,
                                      topk_weights=topk_weights,
                                      topk_ids=topk_ids,
                                      top_k=top_k,
                                      expert_map=expert_map,
                                      ep_group=ep_group)


def fued_moe_with_tp():
    pass