
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



