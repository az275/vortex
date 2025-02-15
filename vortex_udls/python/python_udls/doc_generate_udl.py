#!/usr/bin/env python3
import json
import numpy as np
import pickle
import struct
import torch
import transformers

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic


# class AggBatch:
#     def __init__(self, query_id, doc_ids, distances):
#         self._bytes: np.ndarray = np.ndarray(shape=(0, ), dtype=np.uint8)
#         self._query_ids: list[int] = []
#         self._doc_ids: list[str] = []
#         self._distances: list[float] = []

#     def __str__(self):
#         pass

#     def deserialize_client_notification(blob_data):

#         pass

class ClientNotificationBatcherDeserializer:
    def __init__(self, data: np.ndarray):
        self._bytes = data

        # Define structured dtypes
        header_type = np.dtype([
            ('num_aggregates', np.uint32),
            ('top_k', np.uint32)
        ])
        
        metadata_type = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('text_position', np.uint32),
            ('text_size', np.uint32),
            ('doc_ids_position', np.uint32),
            ('doc_ids_size', np.uint32),
        ])

        # Read header using NumPy view
        header = np.frombuffer(data, dtype=header_type, count=1)[0]
        self.num_aggregates, self.top_k = header['num_aggregates'], header['top_k']

        # Read metadata section using NumPy view 
        metadata_start = header_type.itemsize
        metadata_end = metadata_start + metadata_type.itemsize * self.num_aggregates
        metadata = np.frombuffer(data, dtype=metadata_type, offset=metadata_start, count=self.num_aggregates)

        self.queries = []

        for m in metadata:
            query_id = m['query_id']
            client_id = m['client_id']
            text_position = m['text_position']
            text_size = m['text_size']
            doc_ids_position = m['doc_ids_position']
            doc_ids_size = m['doc_ids_size']

            # use NumPy views to slice text and doc IDs
            text = memoryview(data)[text_position:text_position + text_size]  # Avoids unnecessary string decoding
            doc_ids = np.frombuffer(data, dtype=np.int64, offset=doc_ids_position, count=doc_ids_size // np.dtype(np.int64).itemsize)

            self.queries.append({
                'query_id': query_id,
                'client_id': client_id,
                'text': text,  # Can be decoded only when needed
                'doc_ids': doc_ids  # NumPy array, avoids list conversion
            })

    def get_queries(self, decode_texts=False):
        """
        Returns queries, with an option to decode text strings only when necessary.
        """
        if decode_texts:
            return [{
                'query_id': q['query_id'],
                'client_id': q['client_id'],
                'text': q['text'].tobytes().decode("utf-8"),  # Decode on demand
                'doc_ids': q['doc_ids']
            } for q in self.queries]
        return self.queries

class DocGenerateUDL(UserDefinedLogic):
    """
    This UDL is used to retrieve documents and generate response using LLM.
    """
    
    def load_llm(self,):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        

    def __init__(self,conf_str):
        '''
        Constructor
        '''
        # collect the cluster search result {(query_batch_key,query_count):{query_id: ClusterSearchResults, ...}, ...}
        self.cluster_search_res = {}
        # collect the LLM result per client_batch {(query_batch_key,query_count):{query_id: LLMResult, ...}, ...}
        self.llm_res = {}
        self.conf = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.doc_file_name = './perf_data/miniset/doc_list.pickle'
        self.answer_mapping_file = './perf_data/miniset/answer_mapping.pickle'
        self.doc_list = None
        self.answer_mapping = None
        self.pipeline = None
        self.terminators = None
        print("--- DocGenerateUDL initialized")
    
    
    def _get_doc(self, cluster_id, emb_id):
        """
        Helper method to get the document string in natural language.
        load the documents from disk if not in memory.
        @cluster_id: The id of the KNN cluster where the document falls in.
        @emb_id: The id of the document within the cluster.
        @return: The document string in natural language.
        """
        if self.answer_mapping is None:
            with open(self.answer_mapping_file, "rb") as file:
                self.answer_mapping = pickle.load(file)
        if self.doc_list is None:
            with open(self.doc_file_name, 'rb') as file:
                self.doc_list = pickle.load(file)
        return self.doc_list[self.answer_mapping[cluster_id][emb_id]]
          


    def retrieve_documents(self, search_result):
        """
        @search_result: [(cluster_id, emb_id), ...]
        @return doc_list: [document_1, document_2, ...]
        """     
        doc_list = []
        for cluster_id, emb_id in search_result.items():
            doc_list.append(self._get_doc(cluster_id, emb_id))
        return doc_list


    def llm_generate(self, query_text, doc_list):
        """
        @query: client query
        @doc_list: list of documents
        @return: llm generated response
        """    
        messages = [
            {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(doc_list)},
            {"role": "user", "content": query_text},
        ]
        
        llm_result = self.pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = llm_result[0]["generated_text"][-1]['content']
        print(f"for query:{query_text}")
        print(f"the llm generated response: {response}")
        return response
          
               
     
     
    def ocdpo_handler(self,**kwargs):
        key = kwargs["key"]
        blob = kwargs["blob"]
        print("in DocGenerateUDL ocdpo_handler")
        binary_data = np.frombuffer(blob, dtype=np.uint8)
        deserializer = ClientNotificationBatcherDeserializer(binary_data)
        queries = deserializer.get_queries(decode_texts=True)
        print(f"DocGenerateUDL received {len(queries)} queries")
        for query_id in range(len(queries)):
            print(f"{query_id}: {queries[query_id]}")

          

    def __del__(self):
        pass