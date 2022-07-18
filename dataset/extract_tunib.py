import json
from pathlib import Path
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def write_json(res, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
        f.close()

if __name__ == '__main__':
    fold_data_root=Path('/AGC2021/dataset/noun/fold')
    dst_root=Path('/NounMWP/dataset/fold')
    os.makedirs(dst_root,exist_ok=True)
    for fold in range(5):
        filepath=fold_data_root/('noun_fold%s.json'%fold)

        dataset=load_json(filepath)
        tunib_train={k:v for k, v in dataset['train'].items() if 'tunib' in k}
        tunib_test={k:v for k, v in dataset['test'].items() if 'tunib' in k}
        print(len(tunib_train),len(tunib_test))

        ret={'train':tunib_train, 'test':tunib_test}
        write_json(ret, dst_root/('noun_fold%s.json'%fold))
