import copy
import torch
import torch.nn as nn
import torch.nn.functional as func


class GFRU(nn.Module):
    def __init__(self, hidden_size):
        super(GFRU, self).__init__()

        self.layer_w = nn.Linear(hidden_size, hidden_size)
        self.layer_f = nn.Linear(hidden_size, hidden_size)
        self.layer_w_f = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.layer_w.weight.data.uniform_(-initrange, initrange)
        self.layer_w.bias.data.zero_()
        self.layer_f.weight.data.uniform_(-initrange, initrange)
        self.layer_f.bias.data.zero_()
        self.layer_w_f.weight.data.uniform_(-initrange, initrange)
        self.layer_w_f.bias.data.zero_()

    def forward(self, state_w, state_f):
        state_w_ = self.layer_w(state_w)  # (1, batch_size, rnn_dim) -> ((1, batch_size, rnn_dim))
        state_w_ = self.tanh(state_w_)
        state_f_ = self.layer_f(state_f)  # (1, batch_size, rnn_dim) -> ((1, batch_size, rnn_dim))
        state_f_ = self.tanh(state_f_)
        state_w_f = torch.cat([state_w_, state_f_], 2)  # (1, batch_size, hidden_dim*2)
        gamma = self.layer_w_f(state_w_f)
        gamma = self.sigmoid(gamma)  # (1, batch_size, 1)

        return gamma


class NETE_USER(nn.Module):
    '''
        revise from NETE (multi-task learning)
    '''

    def __init__(self, nuser, nitem, ntoken, nfeature, emsize, rnn_dim, dropout_prob, hidden_size, t, num_layers=2,
                 nsentiment=2):
        super(NETE_USER, self).__init__()

        # ips_embedding
        self.user_embeddings_mlp = nn.Embedding(nuser, emsize)
        self.item_embeddings_mlp = nn.Embedding(nitem, emsize)
        self.feature_embeddings_mlp = nn.Embedding(nfeature, emsize)
        self.softmax = nn.Softmax(dim=0)
        self.t = t

        # confounder1 MLP
        self.confounder1_layer = nn.Sequential(
            nn.Linear(emsize * 2, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, emsize)
        )
        # confounder2 MLP
        self.confounder2_layer = nn.Sequential(
            nn.Linear(emsize * 2, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, emsize)
        )
        # confounder3 MLP
        self.confounder3_layer = nn.Sequential(
            nn.Linear(emsize * 3, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, emsize)
        )

        # model embedding
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.nitem = nitem
        self.nfeature = nfeature

        # text task
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.sentiment_embeddings = nn.Embedding(nsentiment, emsize)

        self.gru_w = nn.GRU(emsize, rnn_dim, dropout=dropout_prob, batch_first=True)
        self.gru_f = nn.GRU(emsize, rnn_dim, dropout=dropout_prob, batch_first=True)
        self.gfru = GFRU(rnn_dim)
        self.predict_linear = nn.Linear(rnn_dim, ntoken)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.tanh = nn.Tanh()
        self.trans_linear = nn.Linear(emsize * 5, rnn_dim)

        # rating task
        self.first_layer = nn.Linear(emsize * 3, hidden_size)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.last_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # treatment prediction
        self.treat1_layer = nn.Sequential(
            nn.Linear(emsize * 2, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, nitem),
            nn.Softmax(dim=1)
        )
        self.treat2_layer = nn.Sequential(
            nn.Linear(emsize * 2, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, nitem),
            nn.Softmax(dim=1)
        )
        self.treat3_layer = nn.Sequential(
            nn.Linear(emsize * 3, hidden_size),
            nn.Dropout(p=dropout_prob),
            nn.ReLU(),
            nn.Linear(hidden_size, ntoken),
            nn.Softmax(dim=1)
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.predict_linear.weight.data.uniform_(-initrange, initrange)
        self.predict_linear.bias.data.zero_()
        self.trans_linear.weight.data.uniform_(-initrange, initrange)
        self.trans_linear.bias.data.zero_()
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

        for module in [self.confounder1_layer, self.confounder2_layer, self.confounder3_layer, self.treat1_layer,
                       self.treat2_layer, self.treat3_layer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    layer.bias.data.zero_()

        self.user_embeddings_mlp.weight.data.uniform_(-initrange, initrange)
        # self.user_embeddings_mlp.weight.data.normal_(mean=0, std=1)
        self.item_embeddings_mlp.weight.data.uniform_(-initrange, initrange)
        # self.item_embeddings_mlp.weight.data.normal_(mean=0, std=1)
        self.feature_embeddings_mlp.weight.data.uniform_(-initrange, initrange)
        # self.feature_embeddings_mlp.weight.data.normal_(mean=0, std=1)

    def encoder(self, user, item, sentiment_index, fea):  # hidden0_init
        u_emb = self.user_embeddings(user)  # (batch_size, emsize)
        i_emb = self.item_embeddings(item)
        sentiment_feature = self.sentiment_embeddings(sentiment_index)
        fea_emb = self.word_embeddings(fea).squeeze()
        z2 = self.confounder2_layer(torch.cat([u_emb, i_emb], dim=1))
        z3 = self.confounder3_layer(torch.cat((u_emb, i_emb, fea_emb), dim=1))
        unit_emb = torch.cat([u_emb, i_emb, sentiment_feature, z2, z3], 1)
        hidden0 = self.tanh(self.trans_linear(unit_emb))
        return hidden0.unsqueeze(0)  # [1, batch_size, rnn_dim]

    def decoder(self, seq, fea,
                new_state):  # seq: (batch_size, seq_len), hidden: (nlayers, batch_size, hidden_size) nlayers=1
        seq_emb = self.word_embeddings(seq)  # (batch_size, seq_len, emsize)
        fea_emb = self.word_embeddings(fea)  # (batch_size, 1, emsize)
        output_w, hidden_w = self.gru_w(seq_emb,
                                        new_state)  # (batch_size, seq_len, hidden_size) vs. (nlayers, batch_size, hidden_size)
        output_f, hidden_f = self.gru_f(fea_emb,
                                        new_state)  # (batch_size, 1, hidden_size) vs. (nlayers, batch_size, hidden_size)
        gamma = self.gfru(hidden_w, hidden_f)  # (1, batch_size, 1)
        new_state = (1.0 - gamma) * hidden_w + gamma * hidden_f  # (1, batch_size, hidden_size)
        decoded = self.predict_linear(new_state.transpose(0, 1))  # (batch_size, 1, ntoken)
        return func.log_softmax(decoded, dim=-1), new_state

    def forward(self, user, item, sentiment_index, seq, fea):
        hidden = self.encoder(user, item, sentiment_index, fea)
        total_word_prob = None
        for id in range(seq.size(1)):
            inputs = seq[:, id:id + 1]
            if id == 0:
                log_word_prob, hidden = self.decoder(inputs, fea, hidden)
                total_word_prob = log_word_prob
            else:
                log_word_prob, hidden = self.decoder(inputs, fea, hidden)
                total_word_prob = torch.cat([total_word_prob, log_word_prob], 1)
        return total_word_prob  # (batch_size, seq_len, ntoken)

    def predict_rating(self, user, item):  # (batch_size,)
        user_emb = self.user_embeddings(user)  # (batch_size, emsize)
        item_emb = self.item_embeddings(item)
        z1 = self.confounder1_layer(torch.cat((user_emb, item_emb), dim=1))
        ui_concat = torch.cat([user_emb, item_emb, z1], 1)  # (batch_size, emsize * 2)
        hidden = self.relu(self.first_layer(ui_concat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.relu(layer(hidden))  # (batch_size, hidden_size)
            hidden = self.dropout(hidden)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)

        return rating  # (batch_size,)

    def predict_puiz1(self, user, item):
        batch_size = user.size(0)

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        z1 = self.confounder1_layer(torch.cat((user_emb_model, item_emb_model), dim=1))

        user_emb = self.user_embeddings_mlp(user)
        user_z1_emb = torch.mul(user_emb, z1)
        item_emb_tot = self.item_embeddings_mlp.weight

        all_score_1 = None
        all_score_2 = None

        score = torch.zeros(batch_size)
        for index in range(batch_size):
            ui_score = torch.mul(user_z1_emb[index], item_emb_tot).sum(1) / self.t
            all_score_1 = ui_score
            ui_score = self.softmax(ui_score)
            all_score_2 = ui_score
            score[index] = ui_score[item[index]]
        return score, all_score_1, all_score_2

    def predict_puiz2(self, user, item):
        batch_size = user.size(0)

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        z2 = self.confounder2_layer(torch.cat((user_emb_model, item_emb_model), dim=1))

        user_emb = self.user_embeddings_mlp(user)
        user_z2_emb = torch.mul(user_emb, z2)
        item_emb_tot = self.item_embeddings_mlp.weight

        score = torch.zeros(batch_size)
        for index in range(batch_size):
            ui_score = torch.mul(user_z2_emb[index], item_emb_tot).sum(1) / self.t
            ui_score = self.softmax(ui_score)
            score[index] = ui_score[item[index]]
        return score

    def predict_puiz3_f(self, user, item, fea, fea_trans):
        batch_size = user.size(0)

        user_emb_model = self.user_embeddings(user)
        item_emb_model = self.item_embeddings(item)
        fea_emb_model = self.word_embeddings(fea).squeeze()
        z3 = self.confounder3_layer(torch.cat((user_emb_model, item_emb_model, fea_emb_model), dim=1))

        user_emb = self.user_embeddings_mlp(user)
        item_emb = self.item_embeddings_mlp(item)
        user_item_emb = torch.mul(user_emb, item_emb)
        user_item_z3_emb = torch.mul(user_item_emb, z3)
        fea_emb_tot = self.feature_embeddings_mlp.weight

        score = torch.zeros(batch_size)
        for index in range(batch_size):
            ui_score = torch.mul(user_item_z3_emb[index], fea_emb_tot).sum(1) / self.t
            ui_score = self.softmax(ui_score)
            score[index] = ui_score[fea_trans[index]]
        return score

    # treatment Prediction
    def predict_treat1(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        z1 = self.confounder1_layer(torch.cat((user_emb, item_emb), dim=1))
        treat1 = self.treat1_layer(torch.cat((user_emb, z1), dim=1))
        return treat1

    def predict_treat2(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        z2 = self.confounder2_layer(torch.cat((user_emb, item_emb), dim=1))
        treat2 = self.treat2_layer(torch.cat((user_emb, z2), dim=1))
        return treat2

    def predict_treat3(self, user, item, fea):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        fea_emb = self.word_embeddings(fea).squeeze()
        z3 = self.confounder3_layer(torch.cat((user_emb, item_emb, fea_emb), dim=1))
        treat3 = self.treat3_layer(torch.cat((user_emb, item_emb, z3), dim=1))
        return treat3
