import pandas
#imorting os and sys to add the path of different site_package\dir to jupyter notebook
import os
import sys
directory_path = os.path.abspath(os.path.join('F:\LLM-project\cuda\Lib\site-packages'))
if directory_path not in sys.path:
    sys.path.append(directory_path)
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=str, required=True, help='Please provide an llms')
args=parser.parse_args()
print(f'-batch_size:{args.batch_size}')
#block_size is the numbre of blocks/list in the stacks
#batch_size is the number of values int the tensor
batch_size=int(args.batch_size)
block_size=128
max_iters=200
#eval_interval=2500
learning_rate=3e-4
eval_iters=20
dropout=0.2
n_embd=384 
n_layer=1
n_head=1

#this checks if gpu is available or not for fast computation
#as cpu performs task in sequential manner which is time consuming for training and testing purposes
#gpu is used to run more than task parrallely
device ='cuda' if torch.cuda.is_available() else 'cpu'

chars=""
with open('open_corpus/vocab.txt','r',encoding='utf-8') as f:
    text=f.read()
    chars=sorted(list(set(text)))
vocab_size=len(chars)

#encoding and decoding
#mapping from string to int
# string_to_int is a dictionnary which is mode of key value pair of char and its index in the chars set
string_to_int={ch:i for i,ch in enumerate(chars)}

#mapping from int to string 
# int_to_String is a dictionnary which is mode of key value pair of index and its value in the chars set
int_to_string={i:ch for i,ch in enumerate(chars)}

#encode is a list  function which input a string and output a list off indexs of the characters in the string 
encode=lambda s:[string_to_int[c] for c in s]

#lambda function takes a list of integers l as input and returns a string by decoding each integer in the input list using the int_to_string dictionary.
decode=lambda l:''.join([int_to_string[i] for i in l])

#memory map for using small snippets of text from a single file of any size 
def random_chunk(split):
    filename="open_corpus/output_train.txt" if split =='train' else "open_corpus/output_val.txt"
    with open(filename,'rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:

            #detemine the file size and a random positoin to start reading
            file_size=len(mm)
            start_pos=random.randint(0,(file_size)-block_size*batch_size)
            #seek ti the random position vand read the block of text
            mm.seek(start_pos)
            block=mm.read(block_size*batch_size-1)
            decoded_block=block.decode('utf-8',errors='ignore').replace('\r','')
            #train and test splits
            data=torch.tensor(encode(decoded_block),dtype=torch.long)
    return data
#get_batch is used select a specific data from the dataset 
def get_batch(split):
    
    #data is the the tensor on which we are working on 
    #we initially divided the dataset into 80:20 into train vs test
    data=random_chunk(split)
    
    #ix is list of random integers of length batch_size from starting to data-block_size
    ix=torch.randint(len(data)-block_size,(batch_size,))
    
    #x is the stack of training dataset of length 8
    x=torch.stack([data[i:i+block_size] for i in ix])
    
    #y is the same stack with one offset for predicting values
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    #x,y shift to gpu(Cuda) if it is available
    x,y=x.to(device), y.to(device)
    
    #print(device)
    #returning pair of stack x and y
    return x,y


#loss estimation 
# This decorator is used before defining the estimate_loss function. 
#It temporarily disables gradient tracking for all the operations inside the function.
@torch.no_grad()

def estimate_loss():
    #out is initialized as an empty dictionary. It will be used to store the estimated losses for the training and validation datasets.
    out={}
    # sets the model into evaluation mode.
    # In evaluation mode, the model behaves differently from training mode, typically disabling features like dropout and batch normalization.
    model.eval()
    
    for split in ['train','val']: # estimate the loss separately for both the training and validation datasets.
        
        losses =torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)

            # (model) is used to compute predictions (logits) and calculate the loss (loss) between the predictions and the target data.
            logits,loss=model(X,Y)

            #This keeps track of the loss for each iteration.
            losses[k]=loss.item()

        # calculates the mean (average) of the losses obtained during those iterations. 
        out[split]=losses.mean()
    model.train()

    #estimated losses for both the training and validation datasets.
    return out

# Single head of self-attention used in multi-head attention mechanisms within transformer models
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        # Key represents the information you want to compare against or use as a reference
        self.key = nn.Linear(n_embd, head_size, bias=False)

        # Query is the information that you are currently processing or seeking to understand better
        self.query = nn.Linear(n_embd, head_size, bias=False)

        # Value is the information associated with the query which provides additional context
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Buffer contains a lower triangular matrix with ones below the main diagonal and zeros above it.
        # This matrix is used for masking during attention computations.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B, T, and C represent the batch size, sequence length, and input dimension
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)

        # The attention mechanism computes the attention weights using the dot product between query and key vectors,
        # which is scaled by a factor of k.shape[-1]**-0.5 (the square root of the key dimension).
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # This masking ensures that the model doesn't attend to future elements in the sequence
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Normalize them and obtain valid attention probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out
        
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()

        # num_heads represents the number of attention heads to use in parallel.
        # head_size is the number of features captured by each attention head.

        # A container for multiple attention heads.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # self.proj is a linear projection layer used to combine the outputs of the individual attention heads.
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Outputs of the attention heads are concatenated along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()

        # A container for defining a sequence of operations in PyTorch.
        self.net = nn.Sequential(
            # Takes an input of dimension n_embd and produces an intermediate output with a dimension that is four times the input dimension
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),

            # Reduces the dimensionality back to the original input dimension
            nn.Linear(4 * n_embd, n_embd),

            # The dropout layer is used for regularization
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Head size is the number of features that each head will be capturing in our multi-head attention
        head_size = n_embd // n_head

        # Self-attention 
        self.sa = MultiHeadAttention(n_head, head_size)

        # The feedforward layer is responsible for capturing complex patterns and features in the data.
        self.ffwd = FeedForward(n_embd)

        # Layer normalization helps stabilize training by normalizing the activations within each layer.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
        
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # This line creates an embedding layer for token embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # This line creates an embedding layer for positional embeddings.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # This line defines a sequence of neural network blocks.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        # This line defines a layer normalization operation. 
        # Layer normalization is used to stabilize and normalize the activations between layers in a neural network.
        self.ln_f = nn.LayerNorm(n_embd)

        # This line defines a linear (fully connected) layer for language modeling.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):

        B,T=index.shape
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(index) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # This line combines the token embeddings and positional embeddings by element-wise addition
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # .shape is used to unpack the items of logits into B, T, C
            # B is for batch, T is for time, C is for the number of classes
            B, T, C = logits.shape            
            # .view is used to pack them alternate of .shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            
            # This function computes the loss between the predicted logits (logits) and the ground truth labels (targets).
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    # Purpose of the generate function: generate a sequence of tokens or indices given an initial context (index)
    # and a maximum number of new tokens (max_new_tokens).
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        # Create a new tensor for the generated sequence
        generated_sequence = index
    
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self.forward(generated_sequence)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]
            
            # Focus only on the last time step
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            num_samples = 1
            index_next = torch.multinomial(probs, num_samples)
    
            # Append sampled index to the running sequence
            generated_sequence = torch.cat((generated_sequence, index_next), dim=1)
    
        return generated_sequence


# Create an instance of GPTLanguageModel named "model"
model = GPTLanguageModel(vocab_size)

#print('loading model parameters..')
#with open('model-01.pkl','rb') as f:
#    model=pickle.load(f)
#print('loaded succes')
# Move the model to GPU if available
m = model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step:{iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model.forward(xb, yb)

    # This line clears (zeros out) the gradients of the model's parameters.
    # Gradients accumulate during each backward pass, so this step ensures that the gradients start fresh for the current batch.
    optimizer.zero_grad(set_to_none=True)

    # This line computes the gradients of the loss with respect to the model's parameters.
    # These gradients are computed to understand how the loss changes as the parameters are adjusted.
    loss.backward()

    
    # This line updates the model's parameters based on the computed gradients and the learning rate (lr).
    # It effectively performs a parameter update step to minimize the loss.
    optimizer.step()

print(loss.item())
with open('model-01.pkl','wb')as f:
    pickle.dump(model,f)
print('model saved')
