"""
This is the basic framework for a hugging face model. Any hugging face model can be passed in (but only bart modles have been tested). 
"""
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BartModel:
    def __init__(self, huggingFaceModel:str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(huggingFaceModel)
        self.model = AutoModel.from_pretrained(huggingFaceModel)
    
    def getTokens(self, text:str):
        return self.tokenizer(text)
    
    def getEmbeddings(self, text:str):
        tokens = self.tokenizer(text, return_tensors='pt')
        # Forward pass through the model
        outputs = self.model(**tokens)
        # Extract embeddings from the output
        word_embeddings = outputs.last_hidden_state
        # If you want embeddings for all tokens:
        all_token_embeddings = word_embeddings.detach().numpy()
        return all_token_embeddings