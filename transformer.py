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
        print('xxx',type(indices),indices)
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


"""Problem (multihead_self_attention): Implement causal multi-head self-attention (5
points)
Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-
tation should accept (at least) the following parameters:
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h. To test your implementation against our
provided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then,
run uv run pytest -k test_multihead_self_attention to test your implementation."""

class MultiHeadSelfAttention(nn.Module):


    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d = d_model // num_heads
        self.WQ = Linear(self.d_model,self.d_model)
        self.WK = Linear(self.d_model,self.d_model)
        self.WV = Linear(self.d_model,self.d_model)
        self.out = Linear(self.d_model,self.d_model)
    
    def forward(self,x):
        ## x ==> B,T,C
        ### 先假装是一个head
        tmp_shape = x.shape
        T = x.shape[-2]
        Q = self.WQ(x) # ...,T,C
        K = self.WK(x) # ...,T,C
        V = self.WV(x) # ...,T,C
        Q = Q.view(-1,T,self.num_heads,self.d).transpose(1,2) # ...,heads,T,d
        K = K.view(-1,T,self.num_heads,self.d).transpose(1,2)
        V = V.view(-1,T,self.num_heads,self.d).transpose(1,2)
        # make mask
        # True False False
        # True True  False
        # True True  True
        mask = torch.ones(T,T)
        mask = torch.tril(mask)
        mask = mask == 1
        # B,heads,T,d
        v = scaled_dot_product_attention(Q,K,V,mask=mask)
        # B,T,C
        v = v.transpose(1,2).contiguous().view(*tmp_shape)
        v = self.out(v)
        return v


class MultiHeadSelfAttentionRope(nn.Module):


    def __init__(self,d_model,num_heads,theta,max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d = d_model // num_heads
        self.WQ = Linear(self.d_model,self.d_model)
        self.WK = Linear(self.d_model,self.d_model)
        self.WV = Linear(self.d_model,self.d_model)
        self.out = Linear(self.d_model,self.d_model)
        self.rope = RotaryPositionalEmbedding(theta=theta,d_k=self.d,max_seq_len=max_seq_len)
    
    def forward(self,x,token_positions):
        ## x ==> B,T,C
        ### 先假装是一个head
        tmp_shape = x.shape
        T = x.shape[-2]
        Q = self.WQ(x) # ...,T,C
        K = self.WK(x) # ...,T,C
        V = self.WV(x) # ...,T,C
        Q = Q.view(-1,T,self.num_heads,self.d).transpose(1,2) # ...,heads,T,d
        K = K.view(-1,T,self.num_heads,self.d).transpose(1,2)
        V = V.view(-1,T,self.num_heads,self.d).transpose(1,2)
        Q = self.rope(Q,token_positions)
        K = self.rope(K,token_positions)
        # make mask
        # True False False
        # True True  False
        # True True  True
        mask = torch.ones(T,T)
        mask = torch.tril(mask)
        mask = mask == 1
        # B,heads,T,d
        v = scaled_dot_product_attention(Q,K,V,mask=mask)
        # B,T,C
        v = v.transpose(1,2).contiguous().view(*tmp_shape)
        v = self.out(v)
        return v



"""Problem (transformer_block): Implement the Transformer block (3 points)
Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
Transformer block should accept (at least) the following parameters.
d_model: int Dimensionality of the Transformer block inputs.
num_heads: int Number of heads to use in multi-head self-attention.
d_ff: int Dimensionality of the position-wise feed-forward inner layer.
To test your implementation, implement the adapter [adapters.run_transformer_block]. Then
run uv run pytest -k test_transformer_block to test your implementation.
Deliverable: Transformer block code that passes the provided tests."""

class TransformerBlock(nn.Module):

    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms1 = RMSNorm(d_model=d_model)
        self.rms2 = RMSNorm(d_model=d_model)
        self.fn = SwiGLU(d_model=d_model,dff=d_ff)
        self.mha = MultiHeadSelfAttentionRope(d_model=d_model,num_heads=num_heads,theta=theta,max_seq_len=max_seq_len)
    
    def forward(self,x):
        T = x.shape[-2]
        token_positions = torch.arange(T)
        x = x + self.mha(self.rms1(x),token_positions)
        x = x + self.fn(self.rms2(x))
        return x
    

"""Problem (transformer_lm): Implementing the Transformer LM (3 points)
Time to put it all together! Implement the Transformer language model as described in §3.1
and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned
construction parameters for the Transformer block, as well as these additional parameters:
vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
embedding matrix.
context_length: int The maximum context length, necessary for determining the dimensionality of
the position embedding matrix.
num_layers: int The number of Transformer blocks to use.
To test your implementation against our provided tests, you will first need to implement the test
adapter at [adapters.run_transformer_lm]. Then, run uv run pytest -k test_transformer_lm
to test your implementation.
Deliverable: A Transformer LM module that passes the above tests."""


class TransformerLM(nn.Module):

    def __init__(self, vocab_size,context_length,num_layers,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length =context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model,num_heads,d_ff,theta,max_seq_len)   for _ in range(self.num_layers)]
        )
        self.rms = RMSNorm(d_model)
        self.embeddings = Embeddings(num_embeddings=vocab_size, embedding_dim=d_model)
        self.linear = Linear(d_model,vocab_size)
    
    def forward(self,x):
        ### x ===> B,T
        x = self.embeddings(x)
        ## B,T,C
        x = self.blocks(x)
        x = self.rms(x)
        x = self.linear(x)
        return x



"""Problem (cross_entropy): Implement Cross entropy
Deliverable: Write a function to compute the cross entropy loss, which takes in predicted logits
(oi) and targets (xi+1) and computes the cross entropy ℓi = − log softmax(oi)[xi+1]. Your function
should handle the following:
• Subtract the largest element for numerical stability.
• Cancel out log and exp whenever possible.
• Handle any additional batch dimensions and return the average across the batch. As with sec-
tion 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropy
to test your implementation."""


def log_softmax(in_features,dim=-1):
    # inputs: Float[Tensor," batch_size vocab_size"]
    max_values = in_features.max(dim=dim,keepdim=True)[0]
    in_features = in_features - max_values
    sums = torch.sum(torch.exp(in_features),dim=-1,keepdim=True)
    sums_log = torch.log(sums)
    return in_features - sums_log

def cross_entropy(inputs , targets):
    # inputs: Float[Tensor," batch_size vocab_size"]
    # targets: Int[Tensor, " batch_size"]
    logs = -log_softmax(inputs,dim=-1)
    # print('yyyy',inputs.shape,targets.unsqueeze(-1).shape,targets,inputs)
    rs = torch.gather(logs,dim=-1,index=targets.unsqueeze(-1))
    return rs.mean()

    

"""Problem (adamw): Implement AdamW (2 points)
Deliverable: Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your
class should take the learning rate α in __init__, as well as the β, ϵ and λ hyperparameters. To help
you keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameter
objects to a dictionary that stores any information you need for that parameter (for AdamW, this would
be the moment estimates). Implement [adapters.get_adamw_cls] and make sure it passes uv run
pytest -k test_adamw."""


class AdamW(torch.optim.Optimizer):

    def __init__(self, params,lr,betas,eps,weight_decay):
        super().__init__(params=params,defaults=dict(
        alpha = lr,
        beta1 = betas[0],
        beta2 = betas[1],
        eplison = eps,
        lamndax = weight_decay
        ))

    
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if 'm' not in self.state[p]:
                        self.state[p]['m'] = torch.zeros_like(p.grad)
                    if 'v' not in self.state[p]:
                        self.state[p]['v'] = torch.zeros_like(p.grad)       
                    if 't' not in self.state[p]:
                        self.state[p]['t'] = 1
                    t = self.state[p]['t']
                    self.state[p]['m'] = group['beta1']  * self.state[p]['m'] + (1-group['beta1']) * p.grad
                    self.state[p]['v'] = group['beta2'] * self.state[p]['v'] + (1-group['beta2']) * (p.grad**2)
                    alpha_t = group['alpha'] * (1-group['beta2']**t)**0.5 / (1-group['beta1']**t)
                    p.data -= alpha_t * self.state[p]['m'] / (torch.sqrt(self.state[p]['v'])+group['eplison'])
                    p.data -= group['alpha'] * group['lamndax'] * p.data
                    self.state[p]['t'] = t + 1
  

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0, eps=1e-9) -> None:
        assert lr > 0, ValueError(f"Invalid learning rate: {lr}")
        self.eps = eps
        super().__init__(
            params,
            {
                "lr": lr,
                "beta1": betas[0],
                "beta2": betas[1],
                "decay": weight_decay,
            }
        )
        # Initialize momentums to zero
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)
        
    def step(self, closure=None) -> None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            decay = group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m")
                v = state.get("v")
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                # import pdb; pdb.set_trace()
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t / (torch.sqrt(v) + self.eps) * m
                p.data -= lr * decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

"""Problem (learning_rate_schedule): Implement cosine learning rate schedule with
warmup
Write a function that takes t, αmax, αmin, Tw and Tc, and returns the learning rate αt according to
the scheduler defined above. Then implement [adapters.get_lr_cosine_schedule] and make sure
it passes uv run pytest -k test_get_lr_cosine_schedule."""


def get_lr_cosine_schedule(t, a_max, a_min, Tw ,Tc):
    if t < Tw:
        return a_max * t / Tw
    elif t <= Tc:
        return a_min + 0.5 * (1 + math.cos(math.pi * (t-Tw)/(Tc-Tw))) * (a_max - a_min)
    return a_min


"""Problem (gradient_clipping): Implement gradient clipping (1 point)
Write a function that implements gradient clipping. Your function should take a list of parameters
and a maximum ℓ2-norm. It should modify each parameter gradient in place. Use ϵ = 10−6 (the
PyTorch default). Then, implement the adapter [adapters.run_gradient_clipping] and make sure
it passes uv run pytest -k test_gradient_clipping."""


def gradient_clipping(params,max_norm,eps=1e-6):
    total_norm = 0
    for param in params:
        if param.grad is not None:
            total_norm += (param.grad.data ** 2).sum()
    total_norm = total_norm  ** 0.5
    rate = max_norm / (total_norm+eps)
    for param in params:
        if param.grad is not None:
            param.grad.data *= rate

"""Problem (data_loading): Implement data loading (2 points)
Deliverable: Write a function that takes a numpy array x (integer array with token IDs), a
batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both ten-
sors should have shape (batch_size, context_length) containing token IDs, and both should be
placed on the requested device. To test your implementation against our provided tests, you will first
need to implement the test adapter at [adapters.run_get_batch]. Then, run uv run pytest -k
test_get_batch to test your implementation"""

import numpy  as np

def get_batch(x, batch_size, context_length, device):
    max_idx = len(x) - context_length - 1 ## 减一位了targets
    print(max_idx)
    inputs = np.zeros((batch_size,context_length))
    targets = np.zeros((batch_size,context_length))

    idxs = np.random.randint(0,max_idx+1,batch_size)

    for b,idx in enumerate(idxs):
        seq = x[idx:idx+context_length+1]
        inputs[b] = seq[:-1]
        targets[b] = seq[1:]
    
    return torch.LongTensor(inputs,device=device), torch.LongTensor(targets,device=device)



"""Problem (checkpointing): Implement model checkpointing (1 point)
Implement the following two functions to load and save checkpoints:
def save_checkpoint(model, optimizer, iteration, out) should dump all the state from the
first three parameters into the file-like object out. You can use the state_dict method of both
the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
have obj be a dictionary, but you can use whatever format you want as long as you can load your
checkpoint later.
This function expects the following parameters:
model: torch.nn.Module
optimizer: torch.optim.Optimizer
iteration: int
out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
def load_checkpoint(src, model, optimizer) should load a checkpoint from src (path or file-
like object), and then recover the model and optimizer states from that checkpoint. Your
function should return the iteration number that was saved to the checkpoint. You can use
torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
load_state_dict method in both the model and optimizers to return them to their previous
states.
This function expects the following parameters:
src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
model: torch.nn.Module
optimizer: torch.optim.Optimizer
Implement the [adapters.run_save_checkpoint] and [adapters.run_load_checkpoint]
adapters, and make sure they pass uv run pytest -k test_checkpointing."""

import os
import torch

from typing import BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    ) -> None: 
    state_dict = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
    )
    torch.save(state_dict, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]




"""Problem (training_together): Put it together (4 points)
Deliverable: Write a script that runs a training loop to train your model on user-provided input.
In particular, we recommend that your training script allow for (at least) the following:
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-eﬀicient loading of training and validation large datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases)"""


