import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos_enc = []

    for i in range(seq_length):
        enc =  [] 
        for j in range(d_model//2):


            enc.append(np.sin(i/10000**(2*j/d_model)))
            enc.append(np.cos(i/10000**((2*j)/d_model)))
        
        pos_enc.append(enc)

    return np.asarray(pos_enc)