import torch
import torch_npu
import os
import argparse

from .random_input import prepare_weights, prepare_inputs
from .ascend_fused_experts import fused_moe
from .parallel_group import init_env, get_ep_group


def benchmark_config(
    num_tokens: int,
    local_num_experts: int,
    global_num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    num_iters: int = 100,
) -> float:
    

    w1, w2 = prepare_weights(num_experts=local_num_experts,
                             intermediate_size=shard_intermediate_size,
                             hidden_size=hidden_size)
    x, gating_output = prepare_inputs(num_iters,
                                      num_tokens,
                                      hidden_size,
                                      global_num_experts)
    input_gating = torch.empty(num_tokens, global_num_experts, dtype=torch.float32)


    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        # router_logits: (num_tokens, n_experts)
        hidden_states = x
        router_logits = input_gating
        is_prefill = False
        top_k = topk

        experts_hidden_states = fused_moe(
            x=hidden_states,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            top_k=top_k,
            ep_group=get_ep_group(),
            renormalize=True,
            use_grouped_topk=False,
            is_prefill=is_prefill,
        )


    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def main():
    use_mix_moe = int(os.environ.get("USE_MIX_MOE", "0"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=2 if use_mix_moe else 1, help="expert parallel size")
    parser.add_argument("--tp", type=int, default=4 if use_mix_moe else 8, help="tensor parallel size")
    args = parser.parse_args()
    
    parallel_config = {
        "ep_size": args.ep,
        "tp_size": args.tp
    }


    config = {
        "num_experts": 128,
        "intermediate_size": 2048
    }
    # Benchmark args
    batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]
    num_iters = 100
    topk = 6
    hidden_size = 2048
    intermediate_size = config["intermediate_size"]

    tp_size = parallel_config["tp_size"]
    ep_size = parallel_config["ep_size"]
    ep_group, tp_group = init_env(backend="hccl")

    shard_intermediate_size = intermediate_size // tp_size
    global_num_experts=config["num_experts"]
    local_num_experts = global_num_experts // ep_size

    # torch.cuda.set_device("cuda:0")
    for bs in batch_sizes:
        avg = benchmark_config(
            num_tokens=bs,
            local_num_experts=local_num_experts,
            global_num_experts=global_num_experts,
            shard_intermediate_size=shard_intermediate_size,
            hidden_size=hidden_size,
            topk=topk,
            num_iters=num_iters
        )
        # TODO print formatted batch_size, num_iters, num_tokens .... and avg_time
        print()



if __name__ == "main":
    main()