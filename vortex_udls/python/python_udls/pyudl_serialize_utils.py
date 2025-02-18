import numpy as np


class AggregateResultBatch:
    
    def __init__(self, data: np.ndarray = None):
        """
        Initializes AggregateResultBatch.
        If data is provided, it automatically deserializes it.
        """
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.queries: list = []

        if data is not None:
            self.deserialize(data)


    def deserialize(self, data: np.ndarray):
        self._bytes = data
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
        num_aggregates = header['num_aggregates']

        # Read metadata section using NumPy view 
        metadata_start = header_type.itemsize
        # metadata_end = metadata_start + metadata_type.itemsize * num_aggregates
        metadata = np.frombuffer(data, dtype=metadata_type, offset=metadata_start, count=num_aggregates)

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
                'text': text,
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
    

class DocGenResult:
    def __init__(self, query_id: int, client_id: int, text: str, context: list):
        self.query_id = query_id
        self.client_id = client_id
        self.text = text
        self.context = context
        self.response = None
    
    def __str__(self):
        return f"LLMResult(query_id={self.query_id}, client_id={self.client_id}, text={self.text}, response={self.response})"




class DocGenResultBatcher:
    def __init__(self, include_context=True):
        self._bytes: np.ndarray = np.array([], dtype=np.uint8)
        self.include_context = include_context
        
        self.doc_gen_results = []
        
        self.full_texts = []
        self.query_ids = []
        self.client_ids = []

    def add_doc_gen_result(self, doc_gen_result: DocGenResult):
        self.doc_gen_results.append(doc_gen_result)

    def serialize_doc_gen_results(self) -> np.ndarray:
        """
        Serializes the batch into a zero-copy NumPy byte array using structured dtype.
        """
        num_results = len(self.doc_gen_results)

        metadata_dtype = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('response_position', np.uint32),
            ('response_length', np.uint32),
        ])

        if self.include_context:
            metadata_dtype = np.dtype(metadata_dtype.descr + [
                ('context_position', np.uint32),
                ('context_length', np.uint32),
            ])

        header_dtype = np.dtype([('count', np.uint32)])

        # Precompute encoded text, response, and context
        encoded_texts = [res.text.encode('utf-8') for res in self.doc_gen_results]
        encoded_responses = [res.response.encode('utf-8') for res in self.doc_gen_results]
        encoded_contexts = [b'\0'.join(doc.encode('utf-8') for doc in res.context) if self.include_context else b'' for res in self.doc_gen_results]

        text_size_total = sum(len(t) for t in encoded_texts)
        response_size_total = sum(len(r) for r in encoded_responses)
        context_size_total = sum(len(c) for c in encoded_contexts) if self.include_context else 0

        metadata_size = num_results * metadata_dtype.itemsize
        header_size = header_dtype.itemsize
        total_size = header_size + metadata_size + text_size_total + response_size_total + (context_size_total if self.include_context else 0)

        # Allocate buffer
        buffer = np.zeros(total_size, dtype=np.uint8)

        # Write header
        np.frombuffer(buffer[:header_size], dtype=header_dtype)['count'] = num_results

        # Compute positions
        metadata_start = header_size
        text_start = metadata_start + metadata_size
        response_start = text_start + text_size_total
        context_start = response_start + response_size_total if self.include_context else 0

        metadata_view = np.frombuffer(buffer[metadata_start:metadata_start + metadata_size], dtype=metadata_dtype)

        text_pos = text_start
        response_pos = response_start
        context_pos = context_start

        # Serialize each result
        for i, (text_bytes, response_bytes, context_bytes) in enumerate(zip(encoded_texts, encoded_responses, encoded_contexts)):
            metadata_entry = [
                self.doc_gen_results[i].query_id, self.doc_gen_results[i].client_id,
                text_pos, len(text_bytes),
                response_pos, len(response_bytes)
            ]

            if self.include_context:
                metadata_entry.extend([context_pos, len(context_bytes)])

            metadata_view[i] = tuple(metadata_entry)

            buffer[text_pos:text_pos + len(text_bytes)] = np.frombuffer(text_bytes, dtype=np.uint8)
            buffer[response_pos:response_pos + len(response_bytes)] = np.frombuffer(response_bytes, dtype=np.uint8)

            if self.include_context and context_bytes:
                buffer[context_pos:context_pos + len(context_bytes)] = np.frombuffer(context_bytes, dtype=np.uint8)

            text_pos += len(text_bytes)
            response_pos += len(response_bytes)
            if self.include_context:
                context_pos += len(context_bytes)

        self._bytes = buffer
        return buffer
    

    def deserialize_doc_gen_results(self, data: np.ndarray):
        """
        Deserializes structured NumPy byte array back into doc_gen_results.
        """
        self._bytes = data

        # Structured dtype
        header_dtype = np.dtype([
            ('count', np.uint32),
        ])
        
        metadata_dtype = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('response_position', np.uint32),
            ('response_length', np.uint32),
        ])

        if self.include_context:
            metadata_dtype = np.dtype(metadata_dtype.descr + [
                ('context_position', np.uint32),
                ('context_length', np.uint32),
            ])

        # Read header
        header_start = 0
        header_end = header_start + header_dtype.itemsize
        num_results = data[header_start:header_end].view(header_dtype)[0]['count']

        metadata_start = header_end
        metadata_end = metadata_start + metadata_dtype.itemsize * num_results

        # Read metadata
        metadata_records = data[metadata_start:metadata_end].view(metadata_dtype)

        self.doc_gen_results = []

        for record in metadata_records:
            query_id = record['query_id']
            client_id = record['client_id']
            text_position = record['text_position']
            text_length = record['text_length']
            response_position = record['response_position']
            response_length = record['response_length']

            text = memoryview(data)[text_position:text_position + text_length]
            response = memoryview(data)[response_position:response_position + response_length]

            context = None
            if self.include_context:
                context_position = record['context_position']
                context_length = record['context_length']
                context = memoryview(data)[context_position:context_position + context_length]

            self.doc_gen_results.append({
                'query_id': query_id,
                'client_id': client_id,
                'text': text,
                'response': response,
                'context': context if self.include_context else None
            })

    def get_doc_gen_results(self, decode_texts=False):
        """
        Returns document generation results, with an option to decode text, response, and context only when necessary.
        """
        if decode_texts:
            return [{
                'query_id': r['query_id'],
                'client_id': r['client_id'],
                'text': r['text'].tobytes().decode("utf-8"),
                'response': r['response'].tobytes().decode("utf-8"),
                'context': r['context'].tobytes().decode("utf-8").split('\0') if (r['context'] and self.include_context) else None
            } for r in self.doc_gen_results]
        
        return self.doc_gen_results

    def serialize_to_concate_strings(self) -> np.ndarray:
        """
        Serializes the batch into a zero-copy NumPy byte array using structured dtype.
        "query is: <text>, response is: <response>, contexts are: <context_1> | <context_2> | ..."
        Stores query, response, and context as a single concatenated string.
        """
        num_results = len(self.doc_gen_results)

        metadata_dtype = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('full_text_position', np.uint32),
            ('full_text_length', np.uint32),
        ])

        header_dtype = np.dtype([('count', np.uint32)])

        # **Precompute concatenated strings** (avoiding redundant operations)
        concatenated_strings = []
        for res in self.doc_gen_results:
            # Format: "query is: ..., response is: ..., contexts are: ..."
            full_text = f"query is: {res.text}, response is: {res.response}"
            if self.include_context and res.context:
                context_str = " | ".join(res.context)
                full_text += f", contexts are: {context_str}"
            
            concatenated_strings.append(full_text.encode('utf-8'))

        # Compute required sizes
        full_text_size_total = sum(len(s) for s in concatenated_strings)
        metadata_size = num_results * metadata_dtype.itemsize
        header_size = header_dtype.itemsize
        total_size = header_size + metadata_size + full_text_size_total

        # Allocate buffer
        buffer = np.zeros(total_size, dtype=np.uint8)

        # Write header
        np.frombuffer(buffer[:header_size], dtype=header_dtype)['count'] = num_results

        # Compute positions
        metadata_start = header_size
        full_text_start = metadata_start + metadata_size

        metadata_view = np.frombuffer(buffer[metadata_start:metadata_start + metadata_size], dtype=metadata_dtype)

        text_pos = full_text_start

        # Serialize each result
        for i, full_text_bytes in enumerate(concatenated_strings):
            metadata_entry = [
                self.doc_gen_results[i].query_id, 
                self.doc_gen_results[i].client_id,
                text_pos, len(full_text_bytes),
            ]

            metadata_view[i] = tuple(metadata_entry)

            buffer[text_pos:text_pos + len(full_text_bytes)] = np.frombuffer(full_text_bytes, dtype=np.uint8)

            text_pos += len(full_text_bytes)

        self._bytes = buffer
        return buffer

    def deserialize_to_concate_strings(self, data: np.ndarray):
        """
        Deserializes structured NumPy byte array back into `full_texts`, `query_ids`, and `client_ids`.
        """
        self._bytes = data

        # Structured dtype
        header_dtype = np.dtype([
            ('count', np.uint32),
        ])
        
        metadata_dtype = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('full_text_position', np.uint32),
            ('full_text_length', np.uint32),
        ])

        # Read header
        header_start = 0
        header_end = header_start + header_dtype.itemsize
        num_results = data[header_start:header_end].view(header_dtype)[0]['count']

        metadata_start = header_end
        metadata_end = metadata_start + metadata_dtype.itemsize * num_results

        # Read metadata
        metadata_records = data[metadata_start:metadata_end].view(metadata_dtype)

        # Clear previous data
        self.full_texts = []
        self.query_ids = []
        self.client_ids = []

        # Extract data into lists
        for record in metadata_records:
            query_id = record['query_id']
            client_id = record['client_id']
            text_position = record['full_text_position']
            text_length = record['full_text_length']

            # Retrieve and decode full text
            full_text = memoryview(data)[text_position:text_position + text_length].tobytes().decode("utf-8")

            self.full_texts.append(full_text)
            self.query_ids.append(query_id)
            self.client_ids.append(client_id)


    


