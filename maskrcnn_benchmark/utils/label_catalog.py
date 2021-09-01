import os

class LabelCatalog(object):
    LABEL_DIR = 'etc/labels'
    LABELS = {
        "eng_cap": "eng_capital.txt",
        "eng_low": "eng_lower.txt",
        "kor_2350": "kor_2350.txt",
        "num": "numbers.txt",
        "symbols": "symbols.txt",
    }

    @staticmethod
    def get(label_files):
        data_dir = LabelCatalog.LABEL_DIR
        characters = ''
        for label_file in label_files:
            label_path = os.path.join(data_dir, LabelCatalog.LABELS[label_file])
            assert os.path.exists(label_path)
            label = open(label_path, encoding='utf-8').readlines()
            characters += ''.join(label).replace('\n','')
        return characters