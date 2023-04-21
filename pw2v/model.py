import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# fast-text implementation in pytorch

class Similarity(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, source_emb, target_emb, **kwargs):
        # source_emb: (batch_size, embedding_dim)
        # target_emb: (batch_size, embedding_dim) or (batch_size, N, embedding_dim)
        # return: (batch_size), similarity scores for each sample in the batch
        # or (batch_size, N), similarity scores for each sample in the batch and each negative sample
        pass

class DotProduct(Similarity):
    def __init__(self, **kwargs):
        super().__init__("dot")

    def forward(self, source_emb, target_emb, **kwargs):
        if target_emb.dim() == 2:
            return (source_emb * target_emb).sum(dim=1)
        elif target_emb.dim() == 3:
            return (source_emb.unsqueeze(1) * target_emb).sum(dim=2)
        else:
            raise ValueError("target_emb must be 2 or 3 dimensional")
        

class Cosine(Similarity):
    def __init__(self, **kwargs):
        super().__init__("cos")

    def forward(self, source_emb, target_emb, **kwargs):
        if target_emb.dim() == 2:
            return F.cosine_similarity(source_emb, target_emb, dim=1)
        elif target_emb.dim() == 3:
            return F.cosine_similarity(source_emb.unsqueeze(1), target_emb, dim=2)
        else:
            raise ValueError("target_emb must be 2 or 3 dimensional")
     
SIMS = {
    "dot": DotProduct,
    "cos": Cosine,
}

class Loss(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name

    def forward(self, positive_sim, negative_sim):
        # positive_sim: (batch_size)
        # negative_sim: (batch_size, negative_samples)
        # return: (batch_size), losses for each sample in the batch
        pass

class MaxMargin(Loss):
    def __init__(self, margin=1.0):
        super().__init__("maxmargin")
        self.margin = margin

    def forward(self, positive_sim, negative_sim):
        # positive_sim: (batch_size)
        # negative_sim: (batch_size, negative_samples)
        # return: (batch_size), losses for each sample in the batch
        return F.relu(self.margin - positive_sim[..., None] + negative_sim)
    
class NegativeSamplingLoss(Loss):
    def __init__(self):
        super().__init__("ns")

    def forward(self, positive_sim, negative_sim):
        # positive_sim: (batch_size)
        # negative_sim: (batch_size, negative_samples)
        # return: (batch_size), losses for each sample in the batch
        log_target = F.logsigmoid(positive_sim)
        sum_log_sampled = F.logsigmoid(-1*negative_sim).sum(dim=1)  # why sum before logsigmoid???.squeeze()
        loss = log_target + sum_log_sampled
        return -1*loss

LOSSES = {
    "maxmargin": MaxMargin,
    "ns": NegativeSamplingLoss,
}

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, similarity:Similarity=DotProduct(), loss:Loss=MaxMargin()):
        super().__init__()
        self.src = nn.Embedding(vocab_size, embedding_dim, sparse=True)   
        self.tgt = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.similarity = similarity
        self.loss = loss
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.src.weight.data.uniform_(-initrange, initrange)
        if isinstance(self.similarity, Similarity):
            self.tgt.weight.data.uniform_(-0, 0)
        else:
            raise ValueError("similarity must be a subclass of Similarity or DistributionalSimilarity")

    def forward(self, u_pos, v_pos, v_neg):

        embed_u = self.src(u_pos)
        embed_v = self.tgt(v_pos)
        neg_embed_v = self.tgt(v_neg)

        score = self.similarity(embed_u, embed_v, source_ids=u_pos)
        neg_score = self.similarity(embed_u, neg_embed_v, source_ids=u_pos)
        loss = self.loss(score, neg_score)

        return loss.mean()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def save(self, path, vocab):
        # save embeddings (parameters of self.src)
        # format: word \t embedding
        with open(path, "w") as f:
            for word, id in vocab.word2idx.items():
                f.write(word + "\t" + " ".join(map(lambda x: f'{x:.5f}', self.src.weight[id].tolist())) + "\n")
        if hasattr(self.similarity, "save"):
            self.similarity.save(path+".sim", vocab)
