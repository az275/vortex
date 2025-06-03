#!/usr/bin/env python3
from derecho.cascade.udl import UserDefinedLogic
import cascade_context
import numpy as np
import json
import re
from derecho.cascade.member_client import ServiceClientAPI

from FlagEmbedding import BGEM3FlagModel, FlagModel

class EncoderUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(EncoderUDL,self).__init__(conf_str)
          self.conf = json.loads(conf_str)
          print(f"EncoderUDL constructor received json configuration: {self.conf}")
          self.capi = ServiceClientAPI()

          self.encoder = FlagModel(
               'BAAI/bge-small-en-v1.5'
          )
          self.centroids_embeddings = np.array([])
          self.emb_dim = 384
          pass

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler
          '''
          value = kwargs["blob"][1]
          input_str = f"String id {value}"
          print(input_str)
          string_embedding = self.encoder.encode(input_str)
          print(f"String embedding: {string_embedding}")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"EncoderUDL destructor")
          pass
