import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if not seqs: return np.empty((0, 0),dtype=np.int_)

    max_len = max_len if max_len else max([len(seq) for seq in seqs])

    new_seqs = []

    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            new_seqs.append(seqs[i][:max_len])
        elif len(seqs[i]) < max_len:
            new_seqs.append( seqs[i] + [pad_value] * (max_len - len(seqs[i])))
        else:
            new_seqs.append( seqs[i])



    return np.array(new_seqs)