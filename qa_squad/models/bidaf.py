import torch
import pickle

import torch.nn.functional as F
from torch import nn
import numpy as np
from loguru import logger

from qa_squad.config.configuration import GLOVE_EMBEDDINGS_100_WORDS_TXT
from qa_squad.models.layers import CharacterEmbeddingLayer, ContextualEmbeddingLayer
from qa_squad.models.tokenizers import BiDAFTokenizer


class BiDAF(nn.Module):

    def __init__(self, char_vocab_dim, emb_dim, char_emb_dim, num_output_channels,
                 kernel_size, ctx_hidden_dim, device, word_tokenizer: BiDAFTokenizer,
                 glove_embeddings: str = GLOVE_EMBEDDINGS_100_WORDS_TXT):
        """
            BiDAF Model for Question Answering.


        Args:
            char_vocab_dim:
            emb_dim:
            char_emb_dim:
            num_output_channels:
            kernel_size:
            ctx_hidden_dim:
            device:
            word_tokenizer:
            glove_embeddings:
        """
        super().__init__()

        self.device = device

        self.word_tokenizer = word_tokenizer
        self.glove_embeddings = glove_embeddings
        self.word_embedding = self.get_glove_embedding()

        self.character_embedding = CharacterEmbeddingLayer(char_vocab_dim, char_emb_dim,
                                                           num_output_channels, kernel_size)

        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim * 2, ctx_hidden_dim)

        self.dropout = nn.Dropout()

        self.similarity_weight = nn.Linear(emb_dim * 6, 1, bias=False)

        self.modeling_lstm = nn.LSTM(emb_dim * 8, emb_dim, bidirectional=True, num_layers=2, batch_first=True,
                                     dropout=0.2)

        self.output_start = nn.Linear(emb_dim * 10, 1, bias=False)

        self.output_end = nn.Linear(emb_dim * 10, 1, bias=False)

        self.end_lstm = nn.LSTM(emb_dim * 2, emb_dim, bidirectional=True, batch_first=True)

    def get_glove_embedding(self):
        glove_dict = {}
        with open(self.glove_embeddings, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        f.close()

        weights_matrix = np.zeros((len(self.word_tokenizer.word2idx), 100))
        words_found = 0
        logger.debug('Loading Glove embeddings.')
        for word, index in self.word_tokenizer.word2idx.items():
            try:
                weights_matrix[index] = glove_dict[word]
                words_found += 1
            except:
                # some word may not be available
                pass
        logger.info(f'Words found: {words_found}')

        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device), freeze=True)

        return embedding

    def forward(self, ctx, ques, char_ctx, char_ques):

        ctx_len = ctx.shape[1]

        ques_len = ques.shape[1]

        ## GET WORD AND CHARACTER EMBEDDINGS

        ctx_word_embed = self.word_embedding(ctx)
        # ctx_word_embed = [bs, ctx_len, emb_dim]

        ques_word_embed = self.word_embedding(ques)
        # ques_word_embed = [bs, ques_len, emb_dim]

        ctx_char_embed = self.character_embedding(char_ctx)
        # ctx_char_embed =  [bs, ctx_len, emb_dim]

        ques_char_embed = self.character_embedding(char_ques)
        # ques_char_embed = [bs, ques_len, emb_dim]

        ## CREATE CONTEXTUAL EMBEDDING

        ctx_contextual_inp = torch.cat([ctx_word_embed, ctx_char_embed], dim=2)
        # [bs, ctx_len, emb_dim*2]

        ques_contextual_inp = torch.cat([ques_word_embed, ques_char_embed], dim=2)
        # [bs, ques_len, emb_dim*2]

        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)
        # [bs, ctx_len, emb_dim*2]

        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)
        # [bs, ques_len, emb_dim*2]

        ## CREATE SIMILARITY MATRIX

        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1, 1, ques_len, 1)
        # [bs, ctx_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]

        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1, ctx_len, 1, 1)
        # [bs, 1, ques_len, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]

        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, emb_dim*2]

        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, emb_dim*6]

        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]

        ## CALCULATE CONTEXT2QUERY ATTENTION

        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]

        c2q = torch.bmm(a, ques_contextual_emb)
        # [bs] ([ctx_len, ques_len] X [ques_len, emb_dim*2]) => [bs, ctx_len, emb_dim*2]

        ## CALCULATE QUERY2CONTEXT ATTENTION

        b = F.softmax(torch.max(similarity_matrix, 2)[0], dim=-1)
        # [bs, ctx_len]

        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]

        q2c = torch.bmm(b, ctx_contextual_emb)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, emb_dim*2]) => [bs, 1, emb_dim*2]

        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, emb_dim*2]

        ## QUERY AWARE REPRESENTATION

        G = torch.cat([ctx_contextual_emb, c2q,
                       torch.mul(ctx_contextual_emb, c2q),
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)

        # [bs, ctx_len, emb_dim*8]

        ## MODELING LAYER

        M, _ = self.modeling_lstm(G)
        # [bs, ctx_len, emb_dim*2]

        ## OUTPUT LAYER

        M2, _ = self.end_lstm(M)

        # START PREDICTION

        p1 = self.output_start(torch.cat([G, M], dim=2))
        # [bs, ctx_len, 1]

        p1 = p1.squeeze()
        # [bs, ctx_len]

        # END PREDICTION

        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        # [bs, ctx_len, 1] => [bs, ctx_len]

        return p1, p2