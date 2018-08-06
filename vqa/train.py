import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def train(model, train_loader, eval_loader, args):
 
    num_epochs = args.epochs
    output = args.output
    optim = torch.optim.Adamax(model.parameters())

    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            pred= model(v, q, a)
            loss = instance_bce_with_logits(pred, a)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            batch_loss = loss.data[0] * v.size(0)
            total_loss += batch_loss 
            train_score += batch_score

            if i % 100 == 0:
                print('epoch %d, iter %d/%d, batch_loss: %.2f, batch_avg_score: %.2f' \
                        %(epoch, i, len(train_loader), batch_loss, batch_score/v.size(0) ))

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        model.eval()
        eval_score, bound = evaluate(model, eval_loader)
        model.train()

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score

def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, q, a in tqdm(dataloader):
        v = v.cuda()
        q = q.cuda()
        pred, _ = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
