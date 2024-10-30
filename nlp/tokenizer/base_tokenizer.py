import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


import spacy
import torch
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path


class NailaTokenizer:
  def __init__(
      self,
      model_name: str = "en_core_web_sm",
      transformer_model: str = "bert-base-uncased",
      sentence_model: str = "all-MiniLM-L6-v2"
  ):
    
    try:
      self.nlp = spacy.load(model_name)
    except OSError:
      spacy.cli.download(model_name)
      self.nlp = spacy.load(model_name)
    
    self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model)

    self.sentence_transformer = SentenceTransformer(sentence_model)

    self.embedding_cache = {}
  
  def process_text(self, text: str) -> Dict:

    doc = self.nlp(text)

    return {
      'tokens': [token.text for token in doc],
      'lemmas': [token.lemma_ for token in doc],
      'pos_tags': [token.pos_ for token in doc],
      'entities': [(ent.text, ent.label_) for ent in doc.ents],
      'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
      'dependencies': [(token.text, token.dep_, token.head.text)
                       for token in doc]
    }
  
  def get_transformer_encoding(
      self,
      text: str,
      max_length: int = 512
  ) -> Dict:

    encoding = self.transformer_tokenizer(
      text,
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
    )

    return {
      'input_ids': encoding['input_ids'],
      'attention_mask': encoding['attention_mask'],
      'tokens': self.transformer_tokenizer.convert_ids_to_tokens(
        encoding['input_ids'][0]
      )
    }
  
  def get_sentence_embedding(self, text: str) -> torch.Tensor:

    if text not in self.embedding_cache:
      embedding = self.sentence_transformer.encode(
        text,
        convert_to_tensor=True
      )
      self.embedding_cache[text] = embedding

    return self.embedding_cache[text]
    
  def batch_process(
      self,
      texts: List[str],
      include_embeddings: bool = False,
      batch_size: Optional[int] = None
  ) -> List[Dict]:
    
    if batch_size:
      results = []

      for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(self._process_batch(batch, include_embeddings))

      return results
    
    return self._process_batch(texts, include_embeddings)
  
  def _process_batch(
    self,
    texts: List[str],
    include_embeddings: bool
  ) -> List[Dict]:
    
    results = []

    for text in texts:
      result = {
        'spacy': self.process_text(text),
        'transformer': self.get_transformer_encoding(text)
      }
          
      if include_embeddings:
        result['embedding'] = self.get_sentence_embedding(text)
          
        results.append(result)
      
    return results
  
  def similarity_score(self, text1: str, text2: str) -> float:
    
    emb1 = self.get_sentence_embedding(text1)
    emb2 = self.get_sentence_embedding(text2)

    return torch.nn.functional.cosine_similarity(
      emb1.unsqueeze(0),
      emb2.unsqueeze(0)
    ).item()
  
  def save_cache(self, path: str):
    cache_path = Path(path)
    torch.save(self.embedding_cache, cache_path)
  
  def load_cache(self, path: str):
    cache_path = Path(path)
    if cache_path.exists():
      self.embedding_cache = torch.load(cache_path)


class TokenizerPipeline:
  def __init__(self, tokenizer: NailaTokenizer):
    self.tokenizer = tokenizer
    self.processors = []
  
  def add_processor(self, func):
    self.processors.append(func)
  
  def process(self, text: str) -> Dict:
    result = {
      'original': text,
      'base_processing': self.tokenizer.process_text(text)
    }

    for processor in self.processors:
      result.update(processor(text))
    
    return result