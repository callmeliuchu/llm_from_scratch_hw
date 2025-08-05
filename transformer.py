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
    

