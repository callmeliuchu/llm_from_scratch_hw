from bpe import Tokenizer
from transformer import *


path = '/Users/liuchu/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt'
tokenizer = Tokenizer(path,100000,['<|endoftext|>'])
tokenizer.train()
# s = "together now bhcsb c       hh"
# ids = tokenizer.encode(s)
# print(tokenizer.decode(ids) == s)

# print('together' in tokenizer.vocab)
# print('together' in tokenizer.merges)
# print(tokenizer.merges)

with open(path,'r') as f:
    content = f.read()

arr = content.split('<|endoftext|>')
for s in arr:
    ids = tokenizer.encode(s)
    print(ids)
    print(tokenizer.decode(ids))
    print(len(ids))
    print('='*40)


ids = tokenizer.encode(arr[0])
print(arr[0])
print('+'*100)




vocab_size = tokenizer.vocab_size
num_layers = 8
d_model = 128
num_heads = 16
d_ff = 128
theta = 1.2
context_length = 50
max_seq_len = context_length
batch_size = 4
device = 'cpu'
lr=1e-3
betas=(0.99,0.9)
eps=1e-6
weight_decay=1e-6
epoch = 500

model = TransformerLM(vocab_size,context_length,num_layers,d_model,num_heads,d_ff,theta,max_seq_len)
optimizer = AdamW(model.parameters(),lr,betas,eps,weight_decay)

def training_together():
    for _ in range(epoch):
        for s in arr:
            ids = tokenizer.encode(s)
            if len(ids) > max_seq_len:
                x,y = get_batch(ids,batch_size=batch_size,context_length=context_length,device=device)
                print(x,y)
                preds = model(x)
                print('hhhh',vocab_size,preds.shape)
                loss = cross_entropy(preds,y)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def generate_text(context):
    conetxt_ids =  torch.LongTensor(tokenizer.encode(context)).unsqueeze(0) # 1,T
    res = []
    for _ in range(100):
        preds = model(conetxt_ids[:,-max_seq_len:]) # 1,T,vocab_size
        next_tokens = torch.argmax(preds,dim=-1)
        conetxt_ids = torch.concat((conetxt_ids,next_tokens),dim=-1)
    
    conetxt_ids = conetxt_ids.flatten().tolist()
    context = tokenizer.decode(conetxt_ids)
    print(context)


training_together()
generate_text('Once upon a time there was a little boy named Ben')