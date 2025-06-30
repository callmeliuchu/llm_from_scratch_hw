s = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

print(s)



def get_doc_words(doc,words):
    for word in re.findall(PAT, doc):
        # print(word)
        if word:
            key = tuple(list(word))
            if key not in words:
                words[key] = 0
            words[key] += 1


# data = {low: 5, lower: 2, widest: 3, newest: 6}

# Merges We first look at every successive pair of bytes and sum the frequency of the words where they
# appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pair ('es')
# and ('st') are tied, so we take the lexicographically greater pair, ('st'). We would then merge the
# pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.
# In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would
# merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the
# sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e',
# 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].
# If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-
# ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].
# With this vocabulary and set of merges, the word newest would tokenize as [ne, west].


def compare(s1,s2):
    for i in range(min(len(s1),len(s2))):
        if ord(s1[i]) > ord(s2[i]):
            return False
    return True

def merge_count(words):
    count= {}
    for word,freq in words.items():
        for i in range(len(word)-1):
            pair = (word[i],word[i+1])
            if pair not in count:
                count[pair] = 0
            count[pair] += freq
    return count

def merge_pair(words,pair):
    ans = {}
    for word,freq in words.items():
        i = 0
        arr = []
        while i < len(word)-1:
            if (word[i],word[i+1]) == pair:
                arr.append(word[i]+word[i+1])
                i += 2
            else:
                arr.append(word[i])
                i += 1
        if i == len(word) - 1:
            arr.append(word[i])
        ans[tuple(arr)] = freq
    return ans


def process(words,merge_times):
    merges = []
    for _ in range(merge_times):
        best_pair = None
        max_count = -1
        count = merge_count(words)
        for pair,freq in count.items():
            if max_count < freq:
                max_count = freq
                best_pair = pair
            elif max_count == freq:
                if compare(''.join(best_pair),''.join(pair)):
                    best_pair = pair
        if best_pair is not None:
            merges.append(best_pair)
            words = merge_pair(words,best_pair)
        else:
            break
    return merges


# Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
# tokenizer. Your BPE training function should handle (at least) the following input parameters:
# input_path: str Path to a text file with BPE tokenizer training data.
# vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
# initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
# special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
# otherwise affect BPE training.
# Your BPE training function should return the resulting vocabulary and merges:
# vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
# lary) to bytes (token bytes).
# merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
# is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
# <token2>. The merges should be ordered by order of creation.

class Tokenizer:

    def __init__(self,input_path,vocab_size,special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}
        self.id2token = {}
        self.merges = []
        self.vocab_size = vocab_size
    
    def train(self):
        for sp in self.special_tokens:
            if sp not in self.vocab:
                self.vocab[sp] = len(self.vocab)
        words = {}
        with open(self.input_path,'r') as f:
            text = f.read()
            for doc in text.split('<|endoftext|>'):
                get_doc_words(doc,words)
        for word in words:
            for c in word:
                if c not in self.vocab:
                    self.vocab[c] = len(self.vocab)
        merge_times = self.vocab_size - len(self.vocab)
        if merge_times > 0:
            self.merges = process(words,merge_times)
        for merge in self.merges:
            s = ''.join(merge)
            if s not in self.vocab:
                self.vocab[s] = len(self.vocab)
        
        for token,_id in self.vocab.items():
            self.id2token[_id] = token
        # print(self.vocab)
        # print(self.merges)
    
    def encode(self,s):
        words = re.findall(PAT, s)
        ans = []
        for word in words:
            word = list(word)
            while True:
                i = 0
                tmp = []
                has_merge = False
                while i < len(word)-1:
                    if (word[i],word[i+1]) in self.merges:
                        tmp.append(word[i]+word[i+1])
                        has_merge = True
                        i += 2
                    else:
                        tmp.append(word[i])
                        i += 1
                if i < len(word):
                    tmp.append(word[i])
                word = tmp
                if not has_merge:
                    break
            ids = [self.vocab[w] for w in word]
            ans.extend(ids)
        return ans
    
    def decode(self,ids):
        return ''.join([self.id2token[_id] for _id in ids])
            

            

            

tokenizer = Tokenizer('/Users/liuchu/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt',100000,['<|endoftext|>'])
tokenizer.train()

ids = tokenizer.encode("together now bhcsb c       hh")
print(tokenizer.decode(ids))

# print('together' in tokenizer.vocab)
# print('together' in tokenizer.merges)
# print(tokenizer.merges)