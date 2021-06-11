import torch
from numpy import load, save
import random
from navec import Navec
from slovnet.model.emb import NavecEmbedding
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from decode_dataset import decode_sentence

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID_DIM = 300
EMB_DIM = 32
MAX_LEN = 20
N_EPOCHS = 20
BATCH_SIZE = 50
DROPOUT = 0
VALID_SIZE = 0.15


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, vocab_size):
        super().__init__()
        self.hid_dim = hid_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = NavecEmbedding(navec)
        self.rnn = torch.nn.GRU(300, hid_dim, num_layers=n_layers, dropout=dropout,
                                batch_first=True,
                                bidirectional=True)

    def forward(self, src_batch):
        src_emb = self.embedding(src_batch)
        outputs, hidden = self.rnn(src_emb)
        summed = torch.sum(hidden, dim=0).unsqueeze(0)

        return outputs, summed


class Decoder(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, vocab_size):
        super().__init__()
        self.output_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = NavecEmbedding(navec)
        self.rnn = torch.nn.GRU(300, hid_dim, num_layers=n_layers,
                                dropout=dropout,
                                batch_first=True)
        self.attn = torch.nn.Linear(hid_dim * 2, MAX_LEN)
        self.attn_combine = torch.nn.Linear(hid_dim * 3, hid_dim, bias=False)
        self.linear = torch.nn.Linear(hid_dim, vocab_size)

    def forward(self, src_batch, hidden, encoder_outputs):
        embedded = self.embedding(src_batch)

        attn_weights = torch.nn.functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        outputs, hidden = self.rnn(output, hidden)
        prediction = self.linear(outputs)

        return prediction, hidden


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, DEVICE):
        super().__init__()
        self.device = DEVICE
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_batch, trg_batch):
        encoder_outputs, encoder_hidden = self.encoder(src_batch)
        prediction, decoder_hidden = self.decoder(trg_batch, encoder_hidden, encoder_outputs)

        return encoder_hidden, prediction


def train(x_train, y_train, seq2seq, cross_entropy, optimizer):
    epoch_loss = 0
    k = 0
    for i in range(0, len(x_train), BATCH_SIZE):
        batch_x = x_train[i: i + BATCH_SIZE]
        batch_y = y_train[i: i + BATCH_SIZE]
        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)

        optimizer.zero_grad()
        encoder_outputs, encoder_hidden = seq2seq.encoder(batch_x)
        predicted_output = list()
        for idx, item in enumerate(batch_y):
            hidden_state = encoder_hidden[:, idx].unsqueeze(0)
            encoder_output = encoder_outputs[idx].squeeze(0)
            seq = list()
            for token in item[:-1]:
                prediction, hidden_state = seq2seq.decoder(token.unsqueeze(0).unsqueeze(0).long(), hidden_state,
                                                           encoder_output)
                seq.append(prediction)

            seq = torch.cat(seq, 1)
            predicted_output.append(seq)
        predicted_output = torch.cat(predicted_output, 0)
        loss = cross_entropy(predicted_output.view(-1, vocab_dim), batch_y[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        k = k + 1

    return epoch_loss / k


def validation(x_valid, y_valid, seq2seq, cross_entropy):
    epoch_loss = 0
    q = 0
    sum_acc = 0
    for i in range(0, len(x_valid), BATCH_SIZE):
        batch_x = x_valid[i: i + BATCH_SIZE]
        batch_y = y_valid[i: i + BATCH_SIZE]
        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)

        encoder_outputs, encoder_hidden = seq2seq.encoder(batch_x)
        predicted_output = list()
        for idx, item in enumerate(batch_y):
            hidden_state = encoder_hidden[:, idx].unsqueeze(0)
            encoder_output = encoder_outputs[idx].squeeze(0)
            token = item[0].unsqueeze(0).unsqueeze(0)
            seq = list()
            for _ in item[:-1]:
                prediction, hidden_state = seq2seq.decoder(token, hidden_state, encoder_output)
                seq.append(prediction)
                token = prediction.argmax(-1)

            seq = torch.cat(seq, 1)
            predicted_output.append(seq)
        predicted_output = torch.cat(predicted_output, 0)
        loss = cross_entropy(predicted_output.view(-1, vocab_dim), batch_y[:, 1:].reshape(-1))

        epoch_loss += loss.item()
        q = q + 1

        outputs = predicted_output[:, :-1].argmax(-1)
        k = 0
        for idx, item in enumerate(outputs):
            final_date = list()
            true_date = list()
            for p in item:
                final_date.append(vocab_y[p])
            for t in batch_y[idx][1:-1]:
                true_date.append(vocab_y[t])
            if final_date == true_date:
                k = k + 1
            else:
                print(f"{true_date} != {final_date}")
        print(f"Accuracy: {k / len(outputs)}")
        sum_acc = sum_acc + (k / len(outputs))
    print(f"Mean Accuracy: {sum_acc / q}")

    return epoch_loss / q


if __name__ == '__main__':
    training_set_splt = list(load('training_set_splt_decode.npy', allow_pickle=True))
    vocab_x = list(load('vocab_x.npy', allow_pickle=True))
    vocab_y = list(load('vocab_y.npy', allow_pickle=True))

    path = '/Users/a18910920/Desktop/normalization1/navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(path)

    random.shuffle(training_set_splt)
    set_size = round(len(training_set_splt) * 0.8)
    test_set = training_set_splt[set_size:]
    training_set_splt = training_set_splt[:set_size]

    item_to_index_x = {key: i for i, key in enumerate(vocab_x)}
    item_to_index_y = {key: i for i, key in enumerate(vocab_y)}
    x = list()
    y = list()
    for pair in training_set_splt:
        w1 = list()
        w2 = list()
        w1.append(vocab_x.index('<sos>'))
        for chr in pair[0]:
            w1.append(item_to_index_x[chr])
        w1.append(vocab_x.index('<eos>'))
        while len(w1) < MAX_LEN:
            w1.append(vocab_x.index('<pad>'))
        x.append(w1)
        w2.append(vocab_y.index('<sos>'))
        for chr in pair[1]:
            w2.append(item_to_index_y[chr])
        w2.append(vocab_y.index('<eos>'))
        while len(w2) < MAX_LEN:
            w2.append(vocab_y.index('<pad>'))
        y.append(w2)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=VALID_SIZE)
    vocab_dim = len(vocab_y)
    pad_idx = vocab_x.index('<pad>')

    writer = SummaryWriter()
    encoder = Encoder(32, HID_DIM, n_layers=1, dropout=DROPOUT, vocab_size=vocab_dim)
    decoder = Decoder(32, HID_DIM, n_layers=1, dropout=DROPOUT, vocab_size=vocab_dim)
    seq2seq = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(seq2seq.parameters())

    # training and validation
    best_loss = 10000
    for epoch in range(N_EPOCHS):
        print(f"Epoch #{epoch}")
        train_loss = train(x_train, y_train, seq2seq, cross_entropy, optimizer)
        valid_loss = validation(x_valid, y_valid, seq2seq, cross_entropy)

        writer.add_scalars("Train Loss vs Valid Loss", {"Train Loss": train_loss, "Valid Loss": valid_loss}, epoch)

        print(f"Epoch Train Loss: {train_loss}")
        print(f"Epoch Valid Loss: {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(seq2seq.encoder.state_dict(), 'encoder.dict')
            torch.save(seq2seq.decoder.state_dict(), 'decoder.dict')
    print(f"Best Loss: {best_loss}")
    writer.close()

    # inference
    x_test = list()
    y_test = list()
    for pair in test_set:
        x_test.append(pair[0])
        y_test.append(pair[1])

    x = list()
    for elem in x_test:
        w = list()
        w.append(vocab_x.index('<sos>'))
        for chr in elem:
            w.append(item_to_index_x[chr])
        w.append(vocab_x.index('<eos>'))
        while len(w) < MAX_LEN:
            w.append(vocab_x.index('<pad>'))
        x.append(w)

    seq2seq.encoder.load_state_dict(torch.load('encoder.dict'))
    seq2seq.decoder.load_state_dict(torch.load('decoder.dict'))

    x = torch.tensor(x)
    encoder_outputs, encoder_hidden = seq2seq.encoder(x)
    predicted_output = list()
    for idx, item in enumerate(x):
        hidden_state = encoder_hidden[:, idx].unsqueeze(0)
        encoder_output = encoder_outputs[idx].squeeze(0)
        token = item[0].unsqueeze(0).unsqueeze(0)
        seq = list()
        for _ in item[:-1]:
            prediction, hidden_state = seq2seq.decoder(token, hidden_state, encoder_output)
            seq.append(prediction)
            token = prediction.argmax(-1)

        seq = torch.cat(seq, 1)
        predicted_output.append(seq)
    predicted_output = torch.cat(predicted_output, 0)

    outputs = predicted_output[:, :-1].argmax(-1)
    k = 0
    for idx, item in enumerate(outputs):
        final_data = list()
        for p in item:
            final_data.append(vocab_y[p])
        decoded = decode_sentence(test_set[idx][0], final_data)
        if decoded == test_set[idx][2]:
            k = k + 1
            print(f"{test_set[idx]} == {decoded}")
        else:
            print(f"\t!!! {test_set[idx]} != {decoded}, {final_data}")

    print(f"Accuracy: {k / len(x_test)}")
