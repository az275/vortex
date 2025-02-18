import json
import queue
import time
import threading
import torch
from transformers import BartForSequenceClassification, BartTokenizer

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic

from pyudl_serialize_utils import DocGenResultBatcher


'''
Used for the ablation study of the pipeline
'''


class QACheckUDL(UserDefinedLogic):
    '''
    AnswerCheckUDL is udl that run NLI model to check if the answer contains harmful information
    '''
    def __init__(self,conf_str):
        super(QACheckUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.tl = TimestampLogger()
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.device_name = 'cuda:0'
        self.device =torch.device(self.device_name)
        self.model_name = "facebook/bart-large-mnli"
        self.nli_tokenizer = None
        self.model = None
        self.hypothesis = "harmful"
        
        self.task_queue = queue.Queue()
        
        self.max_batch_size = 4
        self.min_batch_size = 1
        self.batch_timeout = 1.0
        self.running = True
        self.cond_var = threading.Condition() 
        worker_thread = threading.Thread(target=self.batch_collector, daemon=True)
        worker_thread.start()
        
        # Tracking metrics
        self.processed_tasks = 0
        self.total_latency = 0
        
        
        
    def load_model_toGPU(self):
        '''
        Load model to GPU
        '''
        print(f"[{time.time():.2f}] Loading model to GPU...")
        self.nli_tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForSequenceClassification.from_pretrained(self.model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()
        print(f"[{time.time():.2f}] Model loaded to GPU")


    def batch_collector(self):
        """
        Collects tasks into a queue and process them in batches
        """
        
        while self.running:

            batch = []
            
            start_time_batch = time.time()
            with self.cond_var:
                while self.task_queue.qsize() < self.min_batch_size and self.running:
                    remaining_time = self.batch_timeout - (time.time() - start_time_batch)
                    if remaining_time <= 0:
                        break  # timeout
                    self.cond_var.wait(timeout=remaining_time)  

            if not self.running:
                torch.cuda.synchronize()
                break
            
            while len(batch) < self.max_batch_size:
                try:
                    task = self.task_queue.get_nowait()
                    batch.append(task)
                except queue.Empty:
                    break

            if batch:
                print(f"[{time.time():.2f}] Running inference on batch of size {len(batch)}...")
                batch_latency, true_probs = self.textcheck(batch)
                print(f"[{time.time():.2f}] Batch Latency: {batch_latency:.3f}s ")
                self.send_to_next_udl(true_probs)
                

    def textcheck(self, batch_premise):
        """
        Runs batch text classification
        process batch_premise from the queue
        """
        if self.nli_tokenizer is None:
            self.load_model_toGPU()
        
        print(f"[{time.time():.2f}] textcheck Processing batch of size {len(batch_premise)}...")
        # Note: this step is blocking, doesn't release the GIL  
        inputs = self.nli_tokenizer(batch_premise, [self.hypothesis] * len(batch_premise),
                        return_tensors="pt", padding=True, truncation=True).to(self.device)

        batch_start_time = time.time()

        # CUDA exec, releases GIL
        with torch.no_grad():
            result = self.model(**inputs)

        batch_latency = time.time() - batch_start_time
        self.total_latency += batch_latency

        logits = result.logits
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100 

        # for i, prob in enumerate(true_probs.tolist()):
        #     print(f"[{time.time():.2f}] Premise {i+1}: Probability that the label is true: {prob:.2f}%")
        self.processed_tasks += len(batch_premise)
        
        print(f"[{time.time():.2f}] Processed {self.processed_tasks} tasks")

        return batch_latency, true_probs


    def send_to_next_udl(self, result):
        """
        Send result to next UDL
        """
        # self.capi.put(key,result)
        # print(f"[{time.time():.2f}] Sent result to next UDL: {result}")
        pass



    def ocdpo_handler(self,**kwargs):
        """
        Handles incoming tasks
        """
        key = kwargs["key"]
        blob = kwargs["blob"]
        doc_gen_result_batch = DocGenResultBatcher()
        doc_gen_result_batch.deserialize_to_concate_strings(blob)
        query_list = doc_gen_result_batch.full_texts
        print(f"[{time.time():.2f}] Received query: {query_list}")
        self.textcheck(query_list)
        
        

    def __del__(self):
        pass
