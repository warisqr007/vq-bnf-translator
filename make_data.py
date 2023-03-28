import glob2
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

_, _, files = next(os.walk('/mnt/data1/waris/repo/vq-bnf/translation-all/vq64/ppgs'))

has_bdl = []
for file in files:
    _, spkr, fid = file.split("-")
    fid = fid.split(".")[0]

    if 'ppg-BDL-'+fid+".npy" in files:
        has_bdl.append(file)

print(len(has_bdl))

data_train, data_test, labels_train, labels_test = train_test_split(has_bdl, has_bdl, test_size=0.05, random_state=42)


# train_list = "/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/SV2TTS/synthesizer/train_split.txt"
# dev_list = "/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/SV2TTS/synthesizer/dev_split.txt"

# with open(train_list, encoding="utf-8") as f:
#     train_metadata = [line.strip().split("|") for line in f]

train_list=[]
spk_dict={}

for idx in range(len(data_train)):
    _, spkr, fid = data_train[idx].split("-")
    fid = fid.split(".")[0]

    train_list.append(f'{spkr}/{fid}')
    if spkr not in spk_dict:
        spk_dict[spkr] = []
    spk_dict[spkr].append(f'{fid}')

with open('train_all.txt', mode='wt', encoding='utf-8') as myfile:
    for entry in train_list:
        spkr, fid = entry.split('/')
        rnd_spk_uttr = random.choice(spk_dict[spkr])
        myfile.write(f'{entry}/{rnd_spk_uttr}')
        myfile.write('\n')

# with open(dev_list, encoding="utf-8") as f:
#     dev_metadata = [line.strip().split("|") for line in f]

with open('dev_all.txt', mode='wt', encoding='utf-8') as myfile:
    for idx in range(len(data_test)):
        _, spkr, fid = data_test[idx].split("-")

        fid = fid.split(".")[0]
        myfile.write(f'{spkr}/{fid}/{fid}')
        myfile.write('\n')


# wav_file_list = glob2.glob(f"/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/**/*.wav")

# ids = []
# for t in wav_file_list:
#     spkr = t.split('.')[0].split('/')[-3]
#     fid = t.split('.')[0].split('/')[-1]
#     wav  = t.split('.')[0].split('/')[-2]
#     # with open('/path/to/filename.txt', mode='wt', encoding='utf-8') as myfile:

#     ids.append(f'{spkr}/{fid}')

# ids = np.array(ids)
# np.random.shuffle(ids)

# data_train, data_test, labels_train, labels_test = train_test_split(ids, ids, test_size=0.05, random_state=42)

# with open('train.txt', mode='wt', encoding='utf-8') as myfile:
#     for s in data_train:
#         myfile.write(s)
#         myfile.write('\n')
# with open('dev.txt', mode='wt', encoding='utf-8') as myfile:
#     for s in data_test:
#         myfile.write(s)
#         myfile.write('\n')