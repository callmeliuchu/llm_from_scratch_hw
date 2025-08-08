# Problem (linear): Implementing the linear module (1 point)
# Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear
# transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear
# module, except for not having a bias argument or parameter. We recommend the following interface:
# def __init__(self, in_features, out_features, device=None, dtype=None) Construct a
# linear transformation module. This function should accept the following parameters:
# in_features: int final dimension of the input
# out_features: int final dimension of the output
# device: torch.device | None = None Device to store the parameters on
# dtype: torch.dtype | None = None Data type of the parameters
# def forward(self, x: torch.Tensor) -> torch.Tensor Apply the linear transformation to the
# input.
# Make sure to:
# • subclass nn.Module
# • call the superclass constructor
# • construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in
# an nn.Parameter
# • of course, don’t use nn.Linear or nn.functional.linear
# For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to
# initialize the weights.
# To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter
# should load the given weights into your Linear module. You can use Module.load_state_dict for
# this purpose. Then, run uv run pytest -k test_linear.

from torch import nn
import torch

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(out_features,in_features,device=self.device,dtype=self.dtype))
        self._init_weights(0, float(2 / (in_features + out_features)))
    
    def _init_weights(self, mean, var) -> None:
        nn.init.trunc_normal_(self.weight, mean, var, -3 * var, 3 * var)


    def forward(self,x):
        print('x shape',x.shape)
        print('w shape',self.weight.shape)
        return  x @ self.weight.T


# import torch
# from torch import nn
# import einx

# class Linear(torch.nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         device: torch.device | None = None,
#         dtype: torch.dtype | None = None,
#     ) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
#         self.dtype = dtype
#         self.device = device
#         self._init_weights(0, float(2 / (in_features + out_features)))
    
#     def _init_weights(self, mean, var) -> None:
#         nn.init.trunc_normal_(self.weight, mean, var, -3 * var, 3 * var)

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         return einx.dot("... in_features, out_features in_features -> ... out_features", x, self.weight)






"""3.4.3 Embedding Module
As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs
into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from
torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding
vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
torch.LongTensor of token IDs with shape (batch_size, sequence_length).
Problem (embedding): Implement the embedding module (1 point)
Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an
embedding lookup. Your implementation should follow the interface of PyTorch’s built-in
nn.Embedding module. We recommend the following interface:
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct
an embedding module. This function should accept the following parameters:
num_embeddings: int Size of the vocabulary
19
embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectors
for the given token IDs.
Make sure to:
• subclass nn.Module
• call the superclass constructor
• initialize your embedding matrix as a nn.Parameter
• store the embedding matrix with the d_model being the final dimension
• of course, don’t use nn.Embedding or nn.functional.embedding
Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
initialize the weights.
To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run
uv run pytest -k test_embedding."""



class Embeddings(nn.Module):


    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(num_embeddings,embedding_dim))
        self._init_weights(0, float(2 / (num_embeddings + embedding_dim)))
    
    def _init_weights(self, mean, var) -> None:
        nn.init.trunc_normal_(self.weight, mean, var, -3 * var, 3 * var)
    
    def forward(self,token_ids: torch.Tensor):
        B,T = token_ids.shape
        indices = token_ids.view(-1) ## B * T
        matrix = self.weight[indices] ## B * T , n
        result = matrix.view(B,T,-1)
        return result



"""3.5 Pre-Norm Transformer Block
Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wise
feed-forward network (Vaswani et al., 2017, section 3.1).
In the original Transformer paper, the model uses a residual connection around each of the two sub-layers,
followed by layer normalization. This architecture is commonly known as the “post-norm” Transformer, since
layer normalization is applied to the sublayer output. However, a variety of work has found that moving
layer normalization from the output of each sub-layer to the input of each sub-layer (with an additional
layer normalization after the final Transformer block) improves Transformer training stability [Nguyen and
Salazar, 2019, Xiong et al., 2020]—see Figure 2 for a visual representation of this “pre-norm” Transformer
block. The output of each Transformer block sub-layer is then added to the sub-layer input via the residual
connection (Vaswani et al., 2017, section 5.4). An intuition for pre-norm is that there is a clean “residual
stream” without any normalization going from the input embeddings to the final output of the Transformer,
which is purported to improve gradient flow. This pre-norm Transformer is now the standard used in language
models today (e.g., GPT-3, LLaMA, PaLM, etc.), so we will implement this variant. We will walk through
each of the components of a pre-norm Transformer block, implementing them in sequence.
3.5.1 Root Mean Square Layer Normalization
The original Transformer implementation of Vaswani et al. [2017] uses layer normalization [Ba et al., 2016]
to normalize activations. Following Touvron et al. [2023], we will use root mean square layer normalization
(RMSNorm; Zhang and Sennrich, 2019, equation 4) for layer normalization. Given a vector a ∈ Rdmodel of
activations, RMSNorm will rescale each activation ai as follows:
RMSNorm(ai) = ai
RMS(a) gi, (4)
where RMS(a) =
√ 1
dmodel
∑dmodel
i=1 a2
i + ε. Here, gi is a learnable “gain” parameter (there are d_model such
parameters total), and ε is a hyperparameter that is often fixed at 1e-5.
You should upcast your input to torch.float32 to prevent overflow when you square the input. Overall,
your forward method should look like:
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)
Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
Construct the RMSNorm module. This function should accept the following parameters:
d_model: int Hidden dimension of the model
eps: float = 1e-5 Epsilon value for numerical stability
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, x: torch.Tensor) -> torch.Tensor Process an input tensor of shape
(batch_size, sequence_length, d_model) and return a tensor of the same shape.
Note: Remember to upcast your input to torch.float32 before performing the normalization (and
later downcast to the original dtype), as described above.
To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
run pytest -k test_rmsnorm.
"""

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        # d_model: int Hidden dimension of the model
        # eps: float = 1e-5 Epsilon value for numerical stability
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(self.d_model))
    
    def rms(self,x: torch.Tensor):
        s = (x ** 2).sum(dim=-1,keepdim=True) / x.shape[-1] + self.eps
        return x / (s ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        return self.rms(x) * self.weight
    

"""Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
function and a GLU.
Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
for numerical stability.
You should set dff to approximately 8
3 × dmodel in your implementation, while ensuring that
the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
hardware. To test your implementation against our provided tests, you will need to implement
the test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu to
test your implementation"""


class SwiGLU(nn.Module):

    def __init__(self,d_model,dff):
        super().__init__()
        self.d_model = d_model
        self.dff  = dff
        self.W1 = nn.Parameter(torch.randn(self.dff,self.d_model))
        self.W3 = nn.Parameter(torch.randn(self.dff,self.d_model))
        self.W2 = nn.Parameter(torch.randn(self.d_model,self.dff))

    def silu(self,x):
        return x * torch.sigmoid(x)
    
    def forward(self,x):
        return (self.silu(x @ self.W1.T) * (x @ self.W3.T)) @ self.W2.T
    


"""Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor.
The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct the
RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be inputted
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor Process
an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
that you should tolerate x with an arbitrary number of batch dimensions. You should assume
that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
x along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
pytest -k test_rope."""


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self,theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.devivce = device
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #### token_positions  ====> i
        ###  
        rotate_matrix = torch.zeros(self.d_k,self.d_k)
        # c   -s
        # s   c
        # Seq_length
        # 



import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 检查d_k是否为偶数（旋转需要成对处理）
        assert d_k % 2 == 0, "d_k must be even"
        
        # 预计算频率因子
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # 预计算位置索引
        positions = torch.arange(max_seq_len, device=device).float()
        
        # 计算所有位置的角度
        angles = torch.einsum('i,j->ij', positions, freqs)
        
        # 创建缓存
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量 (..., seq_len, d_k)
        token_positions: 位置张量 (..., seq_len)
        返回: 旋转后的张量 (..., seq_len, d_k)
        """
        # 1. 获取序列长度和批次维度
        seq_len = x.size(-2)
        
        # 2. 根据token_positions获取对应的cos和sin值
        # 展平批次维度以便索引
        flat_positions = token_positions.view(-1)
        cos = self.cos_cache[flat_positions].view(*token_positions.shape, -1)
        sin = self.sin_cache[flat_positions].view(*token_positions.shape, -1)
        
        # 3. 将输入张量分成两部分（偶数和奇数索引）
        x1 = x[..., 0::2]  # 偶数索引: 0, 2, 4, ...
        x2 = x[..., 1::2]  # 奇数索引: 1, 3, 5, ...
        
        # 4. 应用旋转操作
        # 旋转公式:
        # [x1_rot]   [ cosθ  -sinθ ] [x1]
        # [x2_rot] = [ sinθ   cosθ ] [x2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x2 * cos + x1 * sin
        
        # 5. 重新组合旋转后的张量
        # 创建输出张量
        x_rotated = torch.empty_like(x)
        # 将旋转后的值放回偶数位置
        x_rotated[..., 0::2] = x1_rot
        # 将旋转后的值放回奇数位置
        x_rotated[..., 1::2] = x2_rot
        
        return x_rotated
    

    """Deliverable: Write a function to apply the softmax operation on a tensor. Your function should
take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
now have a normalized probability distribution. Use the trick of subtracting the maximum value in
the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
To test your implementation, complete [adapters.run_softmax] and make sure it passes uv run
pytest -k test_softmax_matches_pytorch."""

def softmax(in_features: torch.Tensor,dim):
    new_features = in_features - in_features.max(dim=dim,keepdim=True)[0]
    exp  = torch.exp(new_features)
    sums = torch.sum(exp,dim=dim,keepdim=True)
    probs = exp / sums
    return probs
    

"""Problem (scaled_dot_product_attention): Implement scaled dot-product attention
(5 points)
Deliverable: Implement the scaled dot-product attention function. Your implementation should
handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
(batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
dimensions (if provided). The implementation should return an output with the shape (batch_size,
..., d_v). See section 3.3 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
seq_len). The attention probabilities of positions with a mask value of True should collectively sum
to 1, and the attention probabilities of positions with a mask value of False should be zero.
To test your implementation against our provided tests, you will need to implement the test adapter
at [adapters.run_scaled_dot_product_attention].
uv run pytest -k test_scaled_dot_product_attention tests your implementation on third-order
input tensors, while uv run pytest -k test_4d_scaled_dot_product_attention tests your
implementation on fourth-order input tensors."""


def scaled_dot_product_attention(Q: torch.Tensor,K: torch.Tensor,V: torch.Tensor,mask: torch.Tensor):
    print('Q shape',Q.shape)
    print('K shape',K.shape)
    print('V shape',V.shape)
    d = Q.shape[-1]
    scaled = Q @ K.transpose(-2,-1) / (d ** 0.5)
    print('scaled shape',scaled.shape)
    print('mask shape',mask.shape)
    scaled = scaled.masked_fill_(mask==False,float('-inf'))
    scaled = softmax(scaled,-1)
    return scaled @ V

