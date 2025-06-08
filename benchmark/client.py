#!/usr/bin/env python3
import numpy
from derecho.cascade.external_client import ServiceClientAPI

capi = ServiceClientAPI()
prefix = "/encode/"
subgroup_type = "VolatileCascadeStoreWithStringKey"
subgroup_index = 0
shard_index = 0
num_inputs = 10

for i in range(num_inputs):
    input_str = f"This is string id {i}"
    input_val = bytes(input_str,'utf-8')
    res = capi.put(prefix + f"{i}",
                   input_val,
                   subgroup_type=subgroup_type,
                   subgroup_index=subgroup_index,
                   shard_index=shard_index,
                   message_id=i,
                   trigger=True)
