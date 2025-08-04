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
