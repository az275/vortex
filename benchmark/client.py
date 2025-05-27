#!/usr/bin/env python3
import numpy
from derecho.cascade.external_client import ServiceClientAPI

capi = ServiceClientAPI()
prefix = "/print/"
subgroup_type = "VolatileCascadeStoreWithStringKey"
subgroup_index = 0
shard_index = 0
num_inputs = 100

for i in range(num_inputs):
    res = capi.put(prefix + f"{i}",
                   i.to_bytes(2, 'big'),
                   subgroup_type=subgroup_type,
                   subgroup_index=subgroup_index,
                   shard_index=shard_index,
                   message_id=i,
                   trigger=True)
