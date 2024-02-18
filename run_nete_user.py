import os
import torch
import argparse
import torch.nn as nn
from operator import itemgetter

from model import NETE_USER
from metrics.metrics import rouge_score, bleu_score, root_mean_square_error, mean_absolute_error
from utils import NewDataLoader, PeterBatchify, now_time, ids2tokens, set_seed, get_local_time

parser = argparse.ArgumentParser(description='NETE_D')
# data params
parser.add_argument('--dataset', type=str, default='trip',
                    help='dataset name')
# model params
parser.add_argument('--nlayers', type=int, default=4,
                    help='rating prediction layer number, default=4', )
parser.add_argument('--hidden_size', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--emsize', type=int, default=200,
                    help='embedding dimension of users、 items and words, default=200', )
parser.add_argument('--rnn_dim', type=int, default=256,
                    help='dimension of RNN hidden states, default=256')
parser.add_argument('--rating_reg', type=float, default=1,
                    help='rating regularization rate, default=1')
parser.add_argument('--text_reg', type=float, default=1,
                    help='text regularization rate, default=1')
parser.add_argument('--treat_reg', type=float, default=1,
                    help='treatment regularization rate, default=1')
parser.add_argument('--pui_reg', type=float, default=1,
                    help='pui rate, default=1')
parser.add_argument('--dropout_prob', type=float, default=0.2,
                    help='dropout ratio in RNN, default=0.2')
parser.add_argument('--seq_max_len', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--mean_rating', type=int, default=3,
                    help='distinguish the sentiment')
# running params
parser.add_argument('--alternate_num', type=int, default=1,
                    help='max min alternate_num')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--max_lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--t', type=float, default=1,
                    help='initial template param')
parser.add_argument('--protect_num', type=float, default=1e-7,
                    help='prevent the denominator from being zero')

parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--endure_times', type=int, default=3,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--cuda', type=bool, default=True,
                    help='use CUDA')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='set gpu id')

parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--checkpoint', type=str, default='./nete_d/',
                    help='directory to save the final model')
parser.add_argument('--generated_file_path', type=str, default='_nete_d_generated.txt',
                    help='output file for generated text')
args = parser.parse_args()

data_path = 'dataset/' + args.dataset + '/reviews.pickle'
train_data_path = 'dataset/' + args.dataset + '/train.csv'
valid_data_path = 'dataset/' + args.dataset + '/valid.csv'
test_data_path = 'dataset/' + args.dataset + '/test.csv'
if data_path is None:
    parser.error('--data_path should be provided for loading data')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
set_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')
if args.cuda:
    torch.cuda.set_device(args.gpu_id)

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

model_path = ''
generated_file = args.dataset + args.generated_file_path
prediction_path = os.path.join(args.checkpoint, generated_file)
mf_ui_path = './init_ips/mf_ui_' + args.dataset + '.pt'
mf_fui_path = './init_ips/mf_fui_' + args.dataset + '.pt'

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading dataset: {}'.format(args.dataset))
corpus = NewDataLoader(data_path, train_data_path, valid_data_path, test_data_path, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
user_inter_count = torch.from_numpy(corpus.user_inter_count).to(device)

ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
nfeature = len(corpus.feature_set)
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
pad_idx = word2idx['<pad>']

feature_set = corpus.feature_set
feaID_transform_dict = {}
for new_id, old_id in enumerate(feature_set):
    feaID_transform_dict[old_id] = new_id

print(now_time() + '{}: nuser:{} | nitem:{} | ntoken:{} | nfeature:{}'.format(args.dataset, nuser, nitem, ntokens,
                                                                              nfeature))
print(now_time() + 'trainset:{} | validset:{} | testset:{}'.format(trainset_size, validset_size, testset_size))

train_data = PeterBatchify(corpus.train, word2idx, args.seq_max_len, args.batch_size, shuffle=True)
val_data = PeterBatchify(corpus.valid, word2idx, args.seq_max_len, args.batch_size)
test_data = PeterBatchify(corpus.test, word2idx, args.seq_max_len, args.batch_size)

###############################################################################
# Build the model
###############################################################################
print('=' * 89)
# Load the best pretrained model.
mf_ui = None
with open(mf_ui_path, 'rb') as f:
    mf_ui = torch.load(f, map_location={'cuda:0': 'cuda:2'}).to(device)
for i in mf_ui.parameters():
    i.requires_grad = False
mf_fui = None
with open(mf_fui_path, 'rb') as f:
    mf_fui = torch.load(f).to(device)
for i in mf_fui.parameters():
    i.requires_grad = False

model = NETE_USER(nuser, nitem, ntokens, nfeature, args.emsize, args.rnn_dim, args.dropout_prob, args.hidden_size,
                 args.t, args.nlayers).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx, reduction='none')  # ignore the padding when computing loss
rating_criterion = nn.MSELoss(reduction='none')
CE_criterion = nn.CrossEntropyLoss()

# max_optimizer
max_optimizer = torch.optim.Adam([{'params': _.parameters()} for _ in
                                  [model.user_embeddings_mlp, model.item_embeddings_mlp, model.feature_embeddings_mlp]],
                                 lr=args.max_lr)
max_scheduler = torch.optim.lr_scheduler.StepLR(max_optimizer, 1, gamma=0.90)  # gamma: lr_decay
# min_optimizer
min_optimizer = torch.optim.Adam([{'params': v} for k, v in model.named_parameters() if
                                  'user_embeddings_mlp' not in k and 'item_embeddings_mlp' not in k and 'feature_embeddings_mlp' not in k],
                                 lr=args.lr)
min_scheduler = torch.optim.lr_scheduler.StepLR(min_optimizer, 1, gamma=0.25)  # gamma: lr_decay


###############################################################################
# Train & Evaluate for NETE_D
###############################################################################
def train(data, train_hat_weight_ui, train_hat_weight_ui_f):  # train
    # Turn on training mode which enables dropout.
    model.train()
    rating_loss = 0.
    text_loss = 0.
    total_loss = 0.
    total_sample = 0
    while True:
        user, item, rating, seq, feature, score, index = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)  # (batch_size,)
        rating = rating.to(device)  # (batch_size,)
        seq = seq.to(device)  # (batch_size, tgt_len + 1)
        fea = feature.squeeze()  # (batch_size,)
        fea_trans = itemgetter(*fea.numpy())(feaID_transform_dict)
        fea_trans = torch.as_tensor(fea_trans).to(device)
        feature = feature.to(device)  # (batch_size, 1)

        treat_item = torch.zeros((user.size(0), nitem)).to(device)
        treat_item[torch.arange(user.size(0)), item] = 1.0
        treat_fea = torch.zeros((user.size(0), ntokens)).to(device)
        treat_fea[torch.arange(user.size(0)), feature] = 1.0

        # p_hat
        hat_weight_ui = train_hat_weight_ui[index]
        hat_weight_ui_f = train_hat_weight_ui_f[index]

        # MAX MIN training
        for i in range(args.alternate_num):
            # Max
            max_optimizer.zero_grad()

            # p
            weight_uiz1, all_score_1, all_score_2 = model.predict_puiz1(user, item)
            weight_uiz1 = weight_uiz1.to(device)
            weight_uiz2 = model.predict_puiz2(user, item).to(device)
            weight_uiz3_f = model.predict_puiz3_f(user, item, feature, fea_trans).to(device)
            weight_uiz1_under = weight_uiz1 * user_inter_count[user]
            weight_uiz2_under = weight_uiz2 * user_inter_count[user]
            weight_uiz3f_under = weight_uiz2_under * weight_uiz3_f * 1
            r_weight_uiz1 = torch.reciprocal(weight_uiz1_under + args.protect_num)
            r_weight_uiz3f = torch.reciprocal(weight_uiz3f_under + args.protect_num)
            repeat_r_weight_uiz3f = r_weight_uiz3f.repeat_interleave(16)

            rating_p = model.predict_rating(user, item)  # (batch_size,)
            one = torch.ones_like(rating, dtype=torch.long).to(device)
            zero = torch.zeros_like(rating, dtype=torch.long).to(device)
            sentiment_index = torch.where(rating_p < args.mean_rating, zero, one).to(device)
            log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)
            r_loss = torch.mean(r_weight_uiz1 * rating_criterion(rating_p, rating))
            t_loss = torch.mean(
                repeat_r_weight_uiz3f * text_criterion(log_word_prob.view(-1, ntokens), seq[:, 1:].reshape((-1,))))
            puiz1_loss = torch.sum(torch.abs(weight_uiz1 - hat_weight_ui))
            puiz2_loss = torch.sum(torch.abs(weight_uiz2 - hat_weight_ui))
            puiz3_f_loss = torch.sum(torch.abs(weight_uiz3_f - hat_weight_ui_f))
            robust_loss = args.pui_reg * (puiz1_loss + puiz2_loss + puiz3_f_loss)
            loss = - (args.rating_reg * r_loss + args.text_reg * t_loss) + robust_loss
            loss.backward(retain_graph=True)
            max_optimizer.step()

        # MIN
        min_optimizer.zero_grad()

        # p
        weight_uiz1, all_score_1, all_score_2 = model.predict_puiz1(user, item)
        weight_uiz1 = weight_uiz1.to(device)
        weight_uiz2 = model.predict_puiz2(user, item).to(device)
        weight_uiz3_f = model.predict_puiz3_f(user, item, feature, fea_trans).to(device)
        weight_uiz1_under = weight_uiz1 * user_inter_count[user]
        weight_uiz2_under = weight_uiz2 * user_inter_count[user]
        weight_uiz3f_under = weight_uiz2_under * weight_uiz3_f * 1
        r_weight_uiz1 = torch.reciprocal(weight_uiz1_under + args.protect_num)
        r_weight_uiz3f = torch.reciprocal(weight_uiz3f_under + args.protect_num)
        repeat_r_weight_uiz3f = r_weight_uiz3f.repeat_interleave(16)

        rating_p = model.predict_rating(user, item)  # (batch_size,)
        one = torch.ones_like(rating, dtype=torch.long).to(device)
        zero = torch.zeros_like(rating, dtype=torch.long).to(device)
        sentiment_index = torch.where(rating_p < args.mean_rating, zero, one).to(device)
        log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)

        r_loss = torch.mean(r_weight_uiz1 * rating_criterion(rating_p, rating))
        t_loss = torch.mean(
            repeat_r_weight_uiz3f * text_criterion(log_word_prob.view(-1, ntokens), seq[:, 1:].reshape((-1,))))
        treatment_loss = CE_criterion(model.predict_treat1(user, item), treat_item) \
                         + CE_criterion(model.predict_treat2(user, item), treat_item) \
                         + CE_criterion(model.predict_treat3(user, item, feature), treat_fea)
        loss = args.rating_reg * r_loss + args.text_reg * t_loss + args.treat_reg * treatment_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        min_optimizer.step()

        rating_loss += batch_size * r_loss.item()
        text_loss += batch_size * t_loss.item()
        total_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step == data.total_step:
            break
    return rating_loss / total_sample, text_loss / total_sample, total_loss / total_sample


def evaluate(data):
    # Turn on training mode which enables dropout.
    model.eval()
    rating_loss = 0.
    text_loss = 0.
    total_loss = 0.
    total_sample = 0
    rating_predict = []
    while True:
        user, item, rating, seq, feature, score, index = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)  # (batch_size,)
        rating = rating.to(device)  # (batch_size,)
        seq = seq.to(device)  # (batch_size, tgt_len + 1)
        feature = feature.to(device)  # (batch_size, 1)

        rating_p = model.predict_rating(user, item)  # (batch_size,)
        rating_predict.extend(rating_p.tolist())
        one = torch.ones_like(rating, dtype=torch.long).to(device)
        zero = torch.zeros_like(rating, dtype=torch.long).to(device)
        sentiment_index = torch.where(rating_p < args.mean_rating, zero, one).to(device)

        log_word_prob = model(user, item, sentiment_index, seq[:, :-1], feature)  # (batch_size, tgt_len, ntoken)
        r_loss = torch.mean(rating_criterion(rating_p, rating))
        t_loss = torch.mean(text_criterion(log_word_prob.view(-1, ntokens), seq[:, 1:].reshape((-1,))))
        loss = r_loss + t_loss

        rating_loss += batch_size * r_loss.item()
        text_loss += batch_size * t_loss.item()
        total_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step == data.total_step:
            break
    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)

    return rating_loss / total_sample, text_loss / total_sample, total_loss / total_sample, RMSE, MAE


def generate(data):  # generate explanation & evaluate on metrics
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, feature, score, index = data.next_batch()  # (batch_size, seq_len), data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)
            feature = feature.to(device)

            rating_p = model.predict_rating(user, item)  # (batch_size,)
            rating_predict.extend(rating_p.tolist())
            one = torch.ones_like(rating, dtype=torch.long).to(device)
            zero = torch.zeros_like(rating, dtype=torch.long).to(device)
            sentiment_index = torch.where(rating_p < args.mean_rating, zero, one).to(device)

            inputs = seq[:, :1].to(device)  # (batch_size, 1)
            hidden = None
            ids = inputs
            for idx in range(args.seq_max_len):
                # produce a word at each step
                if idx == 0:
                    hidden = model.encoder(user, item, sentiment_index, feature)
                    log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
                else:
                    log_word_prob, hidden = model.decoder(inputs, feature, hidden)  # (batch_size, 1, ntoken)
                word_prob = log_word_prob.squeeze().exp()  # (batch_size, ntoken)
                inputs = torch.argmax(word_prob, dim=1,
                                      keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                ids = torch.cat([ids, inputs], 1)  # (batch_size, len++)
            ids = ids[:, 1:].tolist()  # remove bos
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    print(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    # bleu
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    # rouge
    text_test = [' '.join(tokens) for tokens in tokens_test]  # 32003
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    # generate
    text_out = ''
    for (real, fake) in zip(text_test, text_predict):  # format: ground_truth|context|explanation
        text_out += '{}\n{}\n\n'.format(real, fake)
    return text_out, RMSE, MAE, BLEU1, BLEU4, ROUGE


###############################################################################
# Loop over epochs.
###############################################################################
print(now_time() + 'NETE_USER learning')
best_val_loss = float('inf')
endure_count = 0

train_hat_weight_ui = mf_ui(train_data.user.to(device), train_data.item.to(device)).to(device)
feature_trans = itemgetter(*train_data.feature.squeeze().numpy())(feaID_transform_dict)
feature_trans = torch.as_tensor(feature_trans).to(device)
train_hat_weight_ui_f = mf_fui(train_data.user.to(device), train_data.item.to(device), feature_trans).to(device)

for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))

    rating_loss, text_loss, train_loss = train(train_data, train_hat_weight_ui, train_hat_weight_ui_f)
    print(now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on train'.format
    (rating_loss, text_loss, train_loss))

    rating_loss, text_loss, val_loss, RMSE, MAE = evaluate(val_data)
    val_loss = round(val_loss, 4)
    print(
        now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} | RMSE {:7.4f} | MAE {:7.4f} on valid'
        .format(rating_loss, text_loss, val_loss, RMSE, MAE))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        saved_model_file = '{}-{}.pt'.format('nete_d', get_local_time())
        model_path = os.path.join(args.checkpoint, saved_model_file)
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
        print(now_time() + 'Save the best model' + model_path)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        max_scheduler.step()
        min_scheduler.step()
        print(now_time() + 'Learning rate1 set to {:2.8f}'.format(max_scheduler.get_last_lr()[0]))
        print(now_time() + 'Learning rate2 set to {:2.8f}'.format(min_scheduler.get_last_lr()[0]))

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
print(now_time() + 'Load the best model' + model_path)

# Run on test data.
rating_loss, text_loss, test_loss, RMSE, MAE = evaluate(test_data)
print(now_time() + 'Run on test set:')
print(now_time() + 'rating loss {:4.4f} | text loss {:4.4f} | total loss {:4.4f} on test'.format(rating_loss, text_loss,
                                                                                                 test_loss))

text_o, test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE = generate(test_data)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))

print('Best val:{:7.4f}'.format(best_val_loss))
print('Best test: RMSE {:7.4f} | MAE {:7.4f}'.format(test_RMSE, test_MAE))
print('Best test: BLEU1 {:7.4f} | BLEU4 {:7.4f}'.format(BLEU1, BLEU4))
for (k, v) in ROUGE.items():
    print('Best test: {} {:7.4f}'.format(k, v))

print(now_time() + 'NETE_USER is OK!')
print('=' * 89)

