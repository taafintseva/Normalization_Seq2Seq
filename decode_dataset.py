from tts_python.normalizer.text_normalizator.text import ordinal_to_words
from tts_python.normalizer.text_normalizator.text import number_to_words
from numpy import save
from numpy import load

CASES = ["nominative", "genitive", "dative", "accusative", "ablative", "prepositional"]
GENDERS = ["masculine", "feminine", "neuter", "plural"]

CASE_GEN = ["nominative masculine", "nominative feminine", "nominative neuter", "nominative plural",
            "genitive masculine", "genitive feminine", "genitive neuter", "genitive plural",
            "dative masculine", "dative feminine", "dative neuter", "dative plural",
            "accusative masculine", "accusative feminine", "accusative neuter", "accusative plural",
            "ablative masculine", "ablative feminine", "ablative neuter", "ablative plural",
            "prepositional masculine", "prepositional feminine", "prepositional neuter", "prepositional plural"]
SAMPLE_SIZE = 25000


def decode_sentence(x, y):
    """
    This function is used for decoding back (to the word format) the sentence which is a result of Seq2Seq model.
    """

    final_arr = list()
    for idx, elem in enumerate(y):
        if elem == '0':
            try:
                final_arr.append(x[idx])
            except:
                return ['error']
        elif elem == '<eos>':
            break
        else:
            num = int(elem)
            try:
                if num > 100:
                    res = CASE_GEN[num - 101].split()
                    word = number_to_words.number_to_words(x[idx], gender=res[1], case=res[0])
                else:
                    res = CASE_GEN[num - 1].split()
                    word = ordinal_to_words.ordinal_to_words(x[idx], gender=res[1], case=res[0], process_leading_zeros=False)
            except:
                return ['error']

            if int(x[idx]) >= 1000 and int(x[idx]) < 2000:
                final_arr.extend(word.lower().split()[1:])
            else:
                final_arr.extend(word.lower().split())
    return final_arr


if __name__ == '__main__':
    """
    New format of dataset:
    0: the word remains unchanged
    1 - 24: number is reformated to the (i - 1) case and gender from CASE_GEN in ordinal format
    101 - 124: number is reformated to the (i - 101) case and gender from CASE_GEN not in ordinal format
    """

    vocab_x = {'<unk>', '<pad>', '<sos>', '<eos>'}
    vocab_y = {'<unk>', '<pad>', '<sos>', '<eos>'}
    training_set_splt = list()
    new_dataset = list()
    training_set = list(load('short_lenta.npy'))

    for idx, pair in enumerate(training_set):
        for jdx, elem in enumerate(pair):
            training_set[idx][jdx] = training_set[idx][jdx].replace('.', ' ')
            training_set[idx][jdx] = training_set[idx][jdx].replace(',', ' ')
            training_set[idx][jdx] = training_set[idx][jdx].replace(':', ' ')
            training_set[idx][jdx] = training_set[idx][jdx].replace(';', ' ')
            training_set[idx][jdx] = training_set[idx][jdx].replace('-', ' ')

        spliten0 = training_set[idx][0].split()
        spliten1 = training_set[idx][1].split()
        training_set_splt.append([spliten1, spliten0])

    save('training_set_splt.npy', training_set_splt)

    nom_mas_101 = 0

    for pair in training_set_splt:
        x = pair[0]
        y = pair[1]
        y_new = list()

        j = 0
        for i, token in enumerate(x):
            if x[i] != y[j]:
                for case in CASES:
                    br = False
                    for gender in GENDERS:
                        try:
                            num_ord = ordinal_to_words.ordinal_to_words(x[i], gender=gender, case=case, process_leading_zeros=False)
                            num_num = number_to_words.number_to_words(x[i], gender=gender, case=case)
                        except:
                            break
                        if y[j] == num_ord.split()[0].lower() and y[j + len(num_ord.split()) - 1] == num_ord.split()[-1].lower():
                            y_new.append(str(CASE_GEN.index(f"{case} {gender}") + 1))
                            j = j + len(num_ord.split())
                            br = True
                            break

                        elif len(num_ord.split()) > 1 and y[j] == num_ord.split()[1].lower() and y[j + len(num_ord.split()) - 2] == num_ord.split()[-1].lower():
                            y_new.append(str(CASE_GEN.index(f"{case} {gender}") + 1))
                            j = j + len(num_ord.split()) - 1
                            br = True
                            break

                        elif y[j] == num_num.split()[0].lower() and y[j + len(num_num.split()) - 1] == num_num.split()[-1].lower():
                            y_new.append(str(CASE_GEN.index(f"{case} {gender}") + 101))
                            j = j + len(num_num.split())
                            br = True
                            break

                        elif len(num_num.split()) > 1 and y[j] == num_num.split()[1].lower() and y[j + len(num_num.split()) - 2] == num_num.split()[-1].lower():
                            y_new.append(str(CASE_GEN.index(f"{case} {gender}") + 101))
                            j = j + len(num_num.split()) - 1
                            br = True
                            break

                    if br:
                        break

            else:
                y_new.append('0')
                j = j + 1

        #get balanced dataset where not as many cases with nominative case and masculine gender
        if '101' in y_new and len(new_dataset) < 7000:
            new_dataset.append([x, y_new, y])
        elif '101' not in y_new and len(new_dataset) < SAMPLE_SIZE:
            new_dataset.append([x, y_new, y])
        vocab_x.update(x)
        vocab_y.update(y_new)

    for i in new_dataset:
        print(i)

    save('vocab_x.npy', list(vocab_x))
    save('vocab_y.npy', list(vocab_y))
    save('training_set_splt_decode.npy', new_dataset)
