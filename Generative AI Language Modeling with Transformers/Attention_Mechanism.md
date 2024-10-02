
one-hot encoded vectors

```python
# Device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
split = 'train'

# Training parameters
learning_rate = 3e-4
# for each time the model will deal with 64 samples
batch_size = 64
# The model will undergo 5000 parameter updates
max_iters = 5000              # Maximum training iterations
# After every 200 iterations of training, the model stops training and is evaluated on the validation set.
eval_interval = 200           # Evaluate model every 'eval_interval' iterations in the training loop
# The model will extract 100 batches of data from the validation set
eval_iters = 100              # When evaluating, approximate loss using 'eval_iters' batches

# Architecture parameters
max_vocab_size = 256          # Maximum vocabulary size
# When using subword segmentation methods (such as BPE, byte pair encoding), because some common words may be broken down into subwords.
# eg: unbelievable may be broken down into "un", "believ", and "able".
vocab_size = max_vocab_size   # Real vocabulary size (e.g. BPE has a variable length, so it can be less than 'max_vocab_size')
block_size = 16               # Context length for predictions
n_embd = 32                   # Embedding size
# If The sentence "The cat sat on the mat" needs to be understood, the first head may focus on the relationship between "cat" and "sat", and the second head may focus on the position relationship between "on" and "mat".
num_heads = 2                 # Number of head in multi-headed attention
n_layer = 2                   # Number of Blocks
ff_scale_factor = 4           # Note: The '4' magic number is from the paper: In equation 2 uses d_model=512, but d_ff=2048
dropout = 0.0                 # Normalization using dropout# 10.788929 M parameters

head_size = n_embd // num_heads
assert (num_heads * head_size) == n_embd
```