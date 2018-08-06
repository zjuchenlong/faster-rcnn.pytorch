import torch
import torch.nn as nn
from attention import SoftAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class Attention_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(Attention_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

def build_attention_model(dataset, args):
    num_hid = args.num_hid
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = SoftAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return Attention_Model(w_emb, q_emb, v_att, q_net, v_net, classifier)

