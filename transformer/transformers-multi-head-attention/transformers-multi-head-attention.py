import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    _,n,dm = Q.shape

    dk = dm//num_heads

    Q @= W_q
    K @= W_k
    V @= W_v

    Qi = np.array([Q[:,:,i:i+dk]  for i in range(0,dm,dk) ])
    Ki = np.array([K[:,:,i:i+dk]  for i in range(0,dm,dk) ])
    Kt = np.transpose(Ki,[0,1,3,2])
    Vi = np.array([V[:,:,i:i+dk]  for i in range(0,dm,dk) ])

    Si = np.matmul(Qi,Kt) / np.sqrt(dk)

    Hi = softmax(Si) @ Vi

    return Hi.transpose(1,2,0,3).reshape(_,n,dm) @ W_o
    

    