from tts_python.normalizer.text_normalizator.text import ordinal_to_words
from tts_python.normalizer.text_normalizator.text import number_to_words
from nmt_utils import load_date
from numpy import save
import random

DATASET_SIZE = 50
FORMATS = ['short',
           'medium',
           'long',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'dd MMMM YYY',
           'd MM YY',
           'd MMMM',
           'd MMMM YYY',
           'dd.MM.YY',
           'd.MM.YY',
           'YYYY',
           'd MMMM',
           'd MMM']
MONTHS_MMMM = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
YEAR_ABBR_CASES = {'nominative': 'год', 'genitive': 'года', 'dative': 'году',
                   'accusative': 'год', 'ablative': 'годом', 'prepositional': 'годе'}


def open_short(date, case):
    if '.' in date:
        arr = date.split('.')
    else:
        arr = date.split(' ')

    day = ordinal_to_words.ordinal_to_words(arr[0], gender='neuter', case=case, process_leading_zeros=False)
    month = int(arr[1])
    month = MONTHS_MMMM[month - 1]
    year_int = arr[2]
    if len(arr[2]) == 2 and int(arr[2]) <= 21:
        year_int = f"20{arr[2]}"
    elif len(arr[2]) == 2 and int(arr[2]) > 21:
        year_int = f"19{arr[2]}"
    year = ordinal_to_words.ordinal_to_words(year_int, case='genitive')
    if int(year_int) < 2000:
        year = year[5:]

    return_str = f"{day} {month} {year} года"
    return return_str.lower()


def open_long(date, case):
    arr = date.split(' ')

    day = ordinal_to_words.ordinal_to_words(arr[0], gender='neuter', case=case, process_leading_zeros=False)
    for m in MONTHS_MMMM:
        if m[:3] == arr[1][:3]:
            month = m
    return_str = f"{day} {month}"

    if len(arr) > 2:
        year_int = arr[2]
        year = ordinal_to_words.ordinal_to_words(year_int, case='genitive')
        if int(year_int) < 2000:
            year = year[5:]
        return_str = f"{return_str} {year}"
    if len(arr) == 4:
        return_str = f"{return_str} года"
    return return_str.lower()


def open_year(date, case):
    year_int = date
    year = ordinal_to_words.ordinal_to_words(date, case=case)
    if int(year_int) < 2000:
        year = year[5:]

    year = f"{year} {YEAR_ABBR_CASES[case]}"

    return year.lower()


def load_human_readable(case: str) -> [str]:
    date_format = random.choice(FORMATS)
    date = load_date(date_format)

    if date_format == 'short' or date_format == 'd MM YY' or date_format == 'dd.MM.YY' or date_format == 'd.MM.YY':
        letter_date = open_short(date[0], case)
    elif date_format == 'YYYY':
        letter_date = open_year(date[0], case)
    else:
        letter_date = open_long(date[0], case)

    return [date[0], letter_date]


if __name__ == '__main__':
    vocab = set()
    sentence_dataset = list()
    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"{date[0]} я пошёл на работу"
        y_sent = f"{date[1]} я пошёл на работу"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('nominative')
        x_sent = f"сегодня {date[0]} и светит солнце"
        y_sent = f"сегодня {date[1]} и светит солнце"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"это произошло {date[0]} в парке напротив главной улицы"
        y_sent = f"это произошло {date[1]} в парке напротив главной улицы"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"они думают, она взяла справку {date[0]}"
        y_sent = f"они думают, она взяла справку {date[1]}"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('nominative')
        x_sent = f"это была среда {date[0]}"
        y_sent = f"это была среда {date[1]}"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('nominative')
        x_sent = f"{date[0]} был одним из самых важных дней в истории"
        y_sent = f"{date[1]} был одним из самых важных дней в истории"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"саша родился в солнечную субботу {date[0]}"
        y_sent = f"саша родился в солнечную субботу {date[1]}"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"эта церковь была построена {date[0]} от рождества христова"
        y_sent = f"эта церковь была построена {date[1]} от рождества христова"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('genitive')
        x_sent = f"этот праздник отмечают начиная с {date[0]}"
        y_sent = f"этот праздник отмечают начиная с {date[1]}"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    for i in range(500):
        date = load_human_readable('ablative')
        x_sent = f"это великое событие датируют {date[0]}"
        y_sent = f"это великое событие датируют {date[1]}"
        sentence_dataset.append([x_sent, y_sent])
        vocab.update(x_sent.split())
        vocab.update(y_sent.split())

    vocab = sorted(vocab)
    vocab.extend(['<unk>', '<pad>', '<sos>', '<eos>'])


    save('dataset.npy', sentence_dataset)
    save('vocab.npy', vocab)
    with open("dataset_text.txt", "w") as txt_file:
        for elem in sentence_dataset:
            txt_file.write(f"{elem[0]}\t-\t{elem[1]}\n")
    with open("vocab_text.txt", "w") as txt_file:
        for idx, elem in enumerate(vocab):
            txt_file.write(f"{idx}: {elem}\n")
