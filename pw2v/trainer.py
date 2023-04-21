import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import WordDataset
from model import FastText
from vocab import Vocab
from torch.autograd import Variable

from utils import print_nearest, test
from logging_utils import Logger

class Trainer:
    def __init__(self, model: FastText, vocab: Vocab, optimizer, name="fasttext", logger: Logger = None):
        self.model = model
        self.vocab = vocab
        self.optimizer = optimizer
        self.name = name
        self.logger = logger

    def train(self, dataloader: DataLoader, epochs: int = 1):
        n_batches = len(dataloader) * epochs
        init_lr = None
        for g in self.optimizer.param_groups:
            init_lr = g['lr']
        cur_lr = init_lr
        
        cur_batch = 0
        for epoch in range(epochs):
            for i, (left, right, negs) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader), mininterval=10)):
                left = Variable(left)
                right = Variable(right)
                negs = Variable(negs)

                left = left.to(self.model.device)
                right = right.to(self.model.device)
                negs = negs.to(self.model.device)

                self.optimizer.zero_grad()
                loss = self.model(left, right, negs, 16)
                loss.backward()
                self.optimizer.step()

                for g in self.optimizer.param_groups:
                        cur_lr = init_lr * (1.0 - cur_batch / n_batches)
                        g['lr'] = cur_lr  # progress to zero
                        
                if i % 1000 == 0:
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}, LR: {cur_lr}")
                    if self.logger is not None:
                        self.logger.log_metrics({"loss": loss.item(), "lr": cur_lr}, step=cur_batch, prefix="inner")
                if i % 10000 == 10000 - 1:
                    self.model.save(f"data/{self.name}_{epoch}.txt", self.vocab)
                cur_batch += 1
            
            save_path = f"data/{self.name}_{epoch}.txt"
            self.model.save(save_path, self.vocab)
            metrics = test(save_path)
            if self.logger is not None:
                self.logger.log_metrics(metrics, step=cur_batch, prefix="metrics")

            # print nearest words
            word = "germany"
            print(f"Nearest words for {word}:")
            print_nearest(self.vocab.idx2word, self.model.src.weight, word, 10)
            



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/text8")
    parser.add_argument("--vocab", type=str, default="data/text8.vocab")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--n_negatives", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--lr", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name", type=str, default="fasttext")
    parser.add_argument("--pickle_path", type=str, default="data/text8.pickle")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    vocab = Vocab.from_file(args.vocab)
    model = FastText(len(vocab), args.embedding_dim)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    dataset = WordDataset(args.file, args.vocab, args.window_size, debug=args.debug, n_negatives=args.n_negatives)
    cpus_available = torch.get_num_threads()
    print(f"Using {cpus_available} workers")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpus_available)

    logger = None
    if args.wandb:
        from logging_utils import WandbLogger
        logger = WandbLogger(project="pw2v", name=args.name)
        logger.log_hparams(args)

    trainer = Trainer(model, vocab, optimizer, name=args.name, logger=logger)
    trainer.train(dataloader, epochs=args.epochs)
        
