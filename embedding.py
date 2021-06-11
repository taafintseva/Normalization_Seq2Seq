import torch

Tx = 35
Ty = 10


def create_one_hot_training(training_set, vocab):
    x_arr = list()
    y_arr = list()

    for data in training_set:
        x = data[0]
        y = data[1]

        x_preproc = [vocab.index('<sos>')]
        y_preproc = [vocab.index('<sos>')]

        for symb in x:
            if symb in vocab:
                x_preproc.append(vocab.index(symb))
            else:
                x_preproc.append(vocab.index('<unk>'))
        x_preproc.append(vocab.index('<eos>'))
        while len(x_preproc) <= Tx:
            x_preproc.append(vocab.index('<pad>'))

        for symb in y:
            y_preproc.append(vocab.index(symb))
        y_preproc.append(vocab.index('<eos>'))

        x_arr.append(x_preproc)
        y_arr.append(y_preproc)

    x_arr = torch.tensor(x_arr)
    x_oh = torch.nn.functional.one_hot(x_arr)

    y_arr = torch.tensor(y_arr)
    y_oh = torch.nn.functional.one_hot(y_arr)

    return x_oh, y_oh


def create_one_hot_test(test_set, vocab):
    x_arr = list()

    for data in test_set:
        x_preproc = [vocab.index('<sos>')]
        for symb in data:
            if symb in vocab:
                x_preproc.append(vocab.index(symb))
            else:
                x_preproc.append(vocab.index('<unk>'))
        x_preproc.append(vocab.index('<eos>'))
        while len(x_preproc) <= Tx:
            x_preproc.append(vocab.index('<pad>'))

        x_arr.append(x_preproc)

    x_arr = torch.tensor(x_arr)
    x_oh = torch.nn.functional.one_hot(x_arr)

    return x_oh


def prediction_vec_to_oh(prediction):
    ind_max = prediction.argmax()
    prediction_oh = torch.zeros(1, 1, 40)
    prediction_oh[:, :, ind_max] = 1

    return prediction_oh, ind_max
