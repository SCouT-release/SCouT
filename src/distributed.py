import os

import torch


def setup_torchrun():
    """Initialize a process launched by torchrun; do nothing for normal Python runs."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU SCouT requires CUDA.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return True


def is_distributed():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def get_rank():
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size():
    return torch.distributed.get_world_size() if is_distributed() else 1


def setup_ddp(rank, world_size, port=12357):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    torch.distributed.barrier()


def cleanup_ddp():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process():
    return get_rank() == 0


def distribute_loader(loader):
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size // torch.distributed.get_world_size(),
        sampler=torch.utils.data.distributed.DistributedSampler(
            loader.dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        ),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )
