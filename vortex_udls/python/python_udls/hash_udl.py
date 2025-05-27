#!/usr/bin/env python3
from derecho.cascade.udl import UserDefinedLogic
import cascade_context
import hashlib
import numpy as np
import json
import re
from derecho.cascade.member_client import ServiceClientAPI

class GenerateHashUDL(UserDefinedLogic):
    def __init__(self,conf_str):
        super(GenerateHashUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        print(f"GenerateHash constructor received json configuration: {self.conf}")
        pass

    def ocdpo_handler(self,**kwargs):
        print(f"I recieved kwargs: {kwargs}")
        blob = kwargs["blob"]
        hash_blob = hashlib.md5(blob).hexdigest()
        print(f"Md5 hash of blob is: {hash_blob}")

    def __del__(self):
        print(f"GenerateHashUDL destructor")
        pass
