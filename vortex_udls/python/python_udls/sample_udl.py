#!/usr/bin/env python3
from derecho.cascade.udl import UserDefinedLogic
import cascade_context
import numpy as np
import json
import re
from derecho.cascade.member_client import ServiceClientAPI

class ConsolePrinterUDL(UserDefinedLogic):
     '''
     ConsolePrinter is the simplest example showing how to use the udl
     '''
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(ConsolePrinterUDL,self).__init__(conf_str)
          self.conf = json.loads(conf_str)
          print(f"ConsolePrinter constructor received json configuration: {self.conf}")
          self.capi = ServiceClientAPI()
          pass

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler
          '''
          key = kwargs["key"]
          value = int(kwargs["blob"][1])
          prefix = "/genhash/"
          subgroup_type = "VolatileCascadeStoreWithStringKey"
          subgroup_index = 0
          shard_index = 1

          new_key = prefix + key
          res = self.capi.put(new_key,
                              value.to_bytes(2, 'big'),
                              subgroup_type=subgroup_type,
                              subgroup_index=subgroup_index,
                              shard_index=shard_index,
                              message_id=int(key))

          print(f"I recieved kwargs: {kwargs}")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"ConsolePrinterUDL destructor")
          pass
