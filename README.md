
# comm-ddp

An easy-to-use communication tools for pytorch-DDP (single-node with multi-GPUs).

- features: launch_DDP, cleanup, rank number, world size, gather, etc.

- Installation
```
pip install comm-ddp
```
- Import
```
from comm_ddp import comm
```

- Launch
```
comm.launch_DDP()
```


- print iff. main_rank
```
comm.mprint("Hello World") # only main_rank would print
```

- rank & world size
```
local_rank = comm.get_local_rank()
word_size = comm.get_world_size()
```

- gather
```
comm.gather()
comm.all_gather()
```
- Demo Usecases: [demo.py](https://github.com/chaoyivision/comm_ddp/blob/main/demo.py) (see some examples).

- Don't forget these:
wrap model with DDP
```
from torch.nn.parallel import DistributedDataParallel as DDP
rank=comm.get_local_rank()
if rank is None:
    # Single GPU
    x = x.to(torch.device(device))
else:
    # DDP launched by torchrun
    x = x.to(rank)
    if isinstance(x, nn.Module):
        debug_find=False # manual set for flexible debugging
        x = DDP(x, device_ids=[rank % torch.cuda.device_count()], find_unused_parameters=debug_find)
```
and launch code via 
```
torchrun --nproc_per_node=4 train.py 
```
