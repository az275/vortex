#!/usr/bin/env python3
import time
import json
import warnings

import numpy as np

from typing import Any
from FlagEmbedding import BGEM3FlagModel

import cascade_context #type: ignore
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger

warnings.filterwarnings("ignore")
from pyudl_serialize_utils import Batch



class EncodeSearchUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self._conf: dict[str, Any] = json.loads(conf_str)
        self._tl = TimestampLogger()
        self._encoder = None
        self._batch = Batch()
        self._batch_id = 0

    def ocdpo_handler(self, **kwargs):
        if self._encoder is None:
            # load encoder when we need it to prevent overloading
            # the hardware during startup
            self._encoder = BGEM3FlagModel(
                model_name_or_path=self._conf["encoder_config"]["model"],
                device=self._conf["encoder_config"]["device"],
                use_fp16=False,
            )

        data = kwargs["blob"]
        
        
        self._batch.deserialize(data)
        
        res: Any = self._encoder.encode(
            self._batch.query_list,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embeddings: np.ndarray = res["dense_vecs"]
        self._batch_id += 1


        # format should be {client}_{batch_id}
        key_str = kwargs["key"]
        output_bytes = self._batch.serialize(query_embeddings)
        cascade_context.emit(key_str, output_bytes, message_id=kwargs["message_id"])
        return None
    
    def __del__(self):
        pass
