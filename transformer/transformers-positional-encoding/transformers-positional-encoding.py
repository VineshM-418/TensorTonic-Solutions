import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length)[:, np.newaxis]

    div_terms = np.exp(
            - np.arange(0,d_model,2) * np.log(10000)/d_model
    )

    pos_enc = np.zeros((seq_length,d_model))

    pos_enc[:, ::2] = np.sin(pos*div_terms)
    pos_enc[:, 1::2] = np.cos(pos*div_terms)

    return pos_enc