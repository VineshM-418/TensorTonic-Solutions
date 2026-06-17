import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token

        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token

        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token

        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token


        unique_words = set() 

        for text in texts: 
            for word in text.split(" "):
                unique_words.add(word.lower())

  
        sorted_unique_words = list(unique_words)
        sorted_unique_words.sort()

        for word in sorted_unique_words:
            self.word_to_id[word] = len(self.word_to_id)
            self.id_to_word[len(self.id_to_word)] = word

        self.vocab_size = len(self.id_to_word)

        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
       
        if not text : return []
        
        text = text.lower()
        text = text.split(" ")

        encoded = []

        for word in text:
            encoded.append(self.word_to_id.get(word,1))
        
        return encoded

        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
    
        return " ".join([self.id_to_word.get(id,self.unk_token) for id in ids])

        
            