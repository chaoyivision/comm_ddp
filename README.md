
# comm-ddp <object https://img.shields.io/pypi/dm/comm-ddp?style=plastic>

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
comm.mprint()
```

- rank & world size
```
comm.get_local_rank()
comm.get_world_size()
```

- gather
```
comm.gather()
comm.all_gather()
```
- Demo Usecases: [demo.py](https://github.com/chaoyivision/comm_ddp/blob/main/demo.py) (see some examples).



