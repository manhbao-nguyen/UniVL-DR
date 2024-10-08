import json
import os
from visual import TSVFile
import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WebQADataset_2(Dataset):
    def __init__(self, args, preprocess_fn, data, docs, captions, shuffle, max_queries = None):
        self.neg_num = args.neg_num
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        if not self.shuffle and args.neg_num == 0:
            self.neg_num = self.img_neg_num + self.txt_neg_num
        self.preprocess_fn = preprocess_fn

        self.img_map = {}
        self.img_tsv = []
        self.docs = docs
        self.captions = captions

        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)
        self.data = data
        if max_queries:
            self.data = self.data[:max_queries]

    def __len__(self):
        return len(self.data)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(Image.open(io.BytesIO(base64.b64decode(img))))
        if self.captions != None:
            cap = self.captions[idx]
            return {'img': img, 'cap': cap, 'idx':idx}
        return {'img': img, 'idx':idx}

    def Collector(self, batch):
        queries = []

        img_inputs = []
        img_dict = {}

        txt_inputs = []
        txt_dict = {}
        cap_inputs = []

        pos_idx = []
        neg_idx = []

        processed_batch = {}

        for qid, example in enumerate(batch):
            if example is None:
                continue
            else:
                queries.append(example['query'])
                if 'pos_imgs' in example:
                    for img in example['pos_imgs']:
                        idx = img['idx']
                        if idx not in img_dict:
                            img_dict[idx] = len(img_inputs)
                            img_inputs.append(img['img'])
                            if 'cap' in img:
                                cap_inputs.append(img['cap'])
                if 'pos_txts' in example:
                    for txt in example['pos_txts']:
                        idx = txt['idx']
                        if idx not in txt_dict:
                            txt_dict[idx] = len(txt_inputs)
                            txt_inputs.append(txt['txt'])

                if 'neg_imgs' in example:
                    for instance in example['neg_imgs']:
                        idx = instance['idx']
                        if idx not in img_dict:
                            img_dict[idx] = len(img_inputs)
                            img_inputs.append(instance['img'])
                            if 'cap' in instance:
                                cap_inputs.append(instance['cap'])
                if 'neg_txts' in example:
                    for instance in example['neg_txts']:
                        idx = instance['idx']
                        if idx not in txt_dict:
                            txt_dict[idx] = len(txt_inputs)
                            txt_inputs.append(instance['txt'])

        for qid, example in enumerate(batch):
            if example is None:
                continue
            else:
                if 'pos_imgs' in example:
                    idxs = [instance['idx'] for instance in example['pos_imgs']]
                    pos_idx.append([img_dict[idx] for idx in idxs])
                if 'pos_txts' in example:
                    idxs = [instance['idx'] for instance in example['pos_txts']]
                    pos_idx.append([txt_dict[idx] + len(img_inputs) for idx in idxs])
                if  ('pos_imgs' in example) and ('pos_txts' in example):
                    print(example)
                    raise AssertionError

                neg_indexes = []
                if 'neg_imgs' in example:
                    idxs = [instance['idx'] for instance in example['neg_imgs']]
                    neg_indexes.extend([img_dict[idx] for idx in idxs])
                    # for instance in example['neg_imgs']:
                    #     idx = instance['idx']
                    #     neg_idx.append(img_dict[idx])
                if 'neg_txts' in example:
                    idxs = [instance['idx'] for instance in example['neg_txts']]
                    neg_indexes.extend([txt_dict[idx] + len(img_inputs) for idx in idxs])
                    # for instance in example['neg_txts']:
                        # idx = instance['idx']
                        # neg_idx.append(txt_dict[idx] + len(img_inputs))
                neg_idx.append(neg_indexes)

        processed_batch['queries'] = clip.tokenize(queries)  # truncate=True
        processed_batch['pos_idx'] = pos_idx
        processed_batch['neg_idx'] = neg_idx

        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)

        if len(cap_inputs) != 0:
            assert len(cap_inputs) == len(img_inputs)
            processed_batch['img_caps'] = clip.tokenize(cap_inputs)  # truncate=True

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = clip.tokenize(txt_inputs)  # truncate=True

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['Q']

        # QUERY
        instance = {'query': query}

        # POSITIVE FACTS
        if len(example['img_posFacts']) != 0:
            if self.shuffle:
                idxs = random.sample(example['img_posFacts'], len(example['img_posFacts']))
            else:
                idxs = example['img_posFacts']
            imgs = [self.encode_img(id) for id in idxs]
            instance["pos_imgs"] = imgs #TODO: add all positive imgs
        elif len(example['txt_posFacts']) != 0:
            if self.shuffle:
                idxs = random.sample(example['txt_posFacts'], len(example['txt_posFacts']))
            else:
                idxs = example['txt_posFacts']
            instance["pos_txts"] = [{'idx': id, 'txt': self.docs[id]} for id in idxs] #TODO: add all positive txts
        else:
            raise ('No positive instance!')

        # NEGATIVE FACTS
        if self.neg_num > 0:
            neg_imgs = []
            neg_txts = []
            if 'all_negFacts' in example:
                neg_idx =  example['all_negFacts']
            else:
                neg_idx = example['img_negFacts'] + example['txt_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_idx)
            neg_idx = neg_idx[:self.neg_num]
            for idx in neg_idx:
                if idx in self.captions:
                    img = self.encode_img(idx)
                    neg_imgs.append(img)
                if idx in self.docs:
                    neg_txts.append({'idx': idx, 'txt': self.docs[idx]})
            if len(neg_imgs) > 0:
                instance["neg_imgs"] = neg_imgs
            if len(neg_txts) > 0:
                instance["neg_txts"] = neg_txts

            return instance

        else:
            img_neg_num = self.img_neg_num
            txt_neg_num = self.txt_neg_num
            if len(example['img_negFacts']) < self.img_neg_num:
                img_neg_num = len(example['img_negFacts'])
                txt_neg_num = self.img_neg_num + self.txt_neg_num - img_neg_num
            elif len(example['txt_negFacts']) < self.txt_neg_num:
                txt_neg_num = len(example['txt_negFacts'])
                img_neg_num = self.img_neg_num + self.txt_neg_num - txt_neg_num

            if img_neg_num > 0:
                neg_imgs = []
                neg_img_idx = example['img_negFacts']
                if self.shuffle:
                    np.random.shuffle(neg_img_idx)
                neg_img_idx = neg_img_idx[:img_neg_num]
                for idx in neg_img_idx:
                    img = self.encode_img(idx)
                    neg_imgs.append(img)
                instance["neg_imgs"] = neg_imgs


            if txt_neg_num > 0:
                neg_txts = []
                neg_txt_idx = example['txt_negFacts']
                if self.shuffle:
                    np.random.shuffle(neg_txt_idx)
                neg_txt_idx = neg_txt_idx[:txt_neg_num]
                for idx in neg_txt_idx:
                    neg_txts.append({'idx': idx, 'txt': self.docs[idx]})
                instance["neg_txts"] = neg_txts

            return instance




def load_file(path, txt=True, img=True):
    data = []
    with open(path) as fin:
        assert (txt or img)
        for line in fin:
            example = json.loads(line.strip())
            txt_negFacts = example['txt_negFacts']
            np.random.shuffle(txt_negFacts)
            example['txt_negFacts'] = txt_negFacts

            img_negFacts = example['img_negFacts']
            np.random.shuffle(img_negFacts)
            example['img_negFacts'] = img_negFacts

            if 'all_negFacts' in example:
                all_negFacts = example['all_negFacts']
                np.random.shuffle(all_negFacts)
                example['all_negFacts'] = all_negFacts

            if txt and len(example['txt_posFacts']) != 0:
                data.append(example)
            if img and len(example['img_posFacts']) != 0:
                data.append(example)
    return data

def load_docs(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            data[did] = example['fact']
    return data

def load_caps(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            data[imgid] = example['caption']
    return data
