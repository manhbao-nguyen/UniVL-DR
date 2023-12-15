import sys
import csv
from tqdm import tqdm
import collections
import gzip
import pickle
import faiss
import os
import logging
import argparse
import json
import os.path as op
import numpy as np
import pytrec_eval
logger = logging.getLogger()
import random
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference
from sklearn.metrics import f1_score, recall_score, precision_score

def get_prediction_first_or_threshold(score, index, threshold):
    return index == 0 or score >= threshold

def get_prediction_two_firsts(index):
    return index == 0 or index == 1

def get_prediction_threshold_only(score, threshold):
    return score >= threshold







if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_embed_path", default="/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/UniVL-DR/exp_3/query_embedding.pkl")
    parser.add_argument("--doc_embed_path", default = "/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/UniVL-DR/exp_3/txt_embedding.pkl")
    parser.add_argument("--img_embed_path", default = "/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/UniVL-DR/exp_3/img_embedding.pkl")
    parser.add_argument("--data_path", default='../data/test.json')
    parser.add_argument("--out_path", default='./')
    parser.add_argument('--use_multihop', action='store_true', default=False)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--topN", type=int, default=100)


    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'evaluation_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.dim)
    all_idx = []

    source_id2embedding = {}

    if args.doc_embed_path:
        logger.info("load data from {}".format(args.doc_embed_path))
        with open(args.doc_embed_path, 'rb') as fin:
            doc_idx, doc_embeds = pickle.load(fin)
            cpu_index.add(np.array(doc_embeds, dtype=np.float32))
            #del doc_embeds
            all_idx.extend(doc_idx)
            for i, id in enumerate(doc_idx):
                source_id2embedding[id] = doc_embeds[i]
            # txt = True

    if args.img_embed_path:
        logger.info("load data from {}".format(args.img_embed_path))
        with open(args.img_embed_path, 'rb') as fin:
            img_idx, img_embeds = pickle.load(fin)
            cpu_index.add(np.array(img_embeds, dtype=np.float32))
            #del img_embeds
            all_idx.extend(img_idx)
            for i, id in enumerate(img_idx):
                source_id2embedding[id] = img_embeds[i]
            # img = True
    def load_file(path):
        result = {}
        with open(path, 'r') as file:
            for line in file:
                dico = json.loads(line)
                result[dico["qid"]] = dico
        return result
    path_test = "/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/data/test.json"
    test_data = load_file(path_test)

    with open(args.query_embed_path, 'rb') as fin:
        logger.info("load data from {}".format(args.query_embed_path))
        query_idx, query_embeds = pickle.load(fin)

    scores = []
    labels = []
    preds = []
    query2result = {}

    for qid, q_embed in zip(query_idx, query_embeds):
        query = test_data[qid]
        imgs = set(query["img_negFacts"]) | set(query["img_posFacts"])
        texts = set(query["txt_posFacts"]) | set(query["txt_negFacts"])
        pos_imgs = set(query["img_posFacts"])
        pos_texts = set(query["txt_posFacts"])

        #all_idx = []

        # cpu_index = faiss.IndexFlatIP(args.dim)
        # for doc_id, doc_embed in zip(doc_idx, doc_embeds):
        #     if doc_id in texts:
        #         cpu_index.add(np.array(doc_embed, dtype=np.float32).reshape((1,-1)))
        #         all_idx.append(doc_id)
        #
        # for img_id, img_embed in zip(img_idx, img_embeds):
        #     if img_id in imgs:
        #         cpu_index.add(np.array(img_embed, dtype=np.float32).reshape((1,-1)))
        #         all_idx.append(img_id)

        query_embed = np.array(q_embed, dtype=np.float32).reshape((1,-1))
        D, I = cpu_index.search(query_embed, len(all_idx))
        query2result[qid] = {}
        if not args.use_multihop:
            for i in range(len(D[0])):
                score = D[0][i]
                # def get_prediction_first_or_threshold(score, index, threshold):
                #     return index == 0 or score >= threshold
                #
                # def get_prediction_two_firsts(index):
                #     return index == 0 or index == 1
                #
                # def get_prediction_threshold_only(score, threshold):
                #     return score >= threshold

                if get_prediction_first_or_threshold(score, i, 0.71):
                    preds.append(1)
                    query2result[qid]["pos"] = query2result[qid].get("pos", []) + [all_idx[I[0][i]]]
                else:
                    preds.append(0)
                scores.append(score)

            for id in I[0]:
                if all_idx[id] in pos_imgs or all_idx[id] in pos_texts:
                    labels.append(1)
                    query2result[qid]["true"] = query2result[qid].get("true", []) + [all_idx[id]]
                else:
                    labels.append(0)

        else:
            last_query_embed = query_embed
            score = D[0][0]
            scores.append(score)
            keep_searching = None
            last_pos_source_id = all_idx[I[0][0]]
            if get_prediction_threshold_only(score, 0.65):
                last_pos_source_id = all_idx[I[0][0]]
                preds.append(1)
                query2result[qid]["pos"] = query2result[qid].get("pos", []) + [last_pos_source_id]
                keep_searching = True
            else:
                preds.append(0)
                keep_searching = False

            if last_pos_source_id in pos_imgs or  last_pos_source_id in pos_texts:
                labels.append(1)
                query2result[qid]["true"] = query2result[qid].get("true", []) + [last_pos_source_id]
            else:
                labels.append(0)

            if keep_searching:
                last_pos_source_embedding = np.array(source_id2embedding[last_pos_source_id], dtype=np.float32).reshape((1,-1))
                last_query_embed += last_pos_source_embedding
                D_1, I_1 = cpu_index.search(last_query_embed, len(all_idx))
                score = D_1[0][0]
                if last_pos_source_id == all_idx[I_1[0][0]]:
                    continue
                else:
                    last_pos_source_id = all_idx[I_1[0][0]]
                    scores.append(score)
                    if get_prediction_threshold_only(score, 0.65):
                        preds.append(1)
                        query2result[qid]["pos"] = query2result[qid].get("pos", []) + [last_pos_source_id]
                    else:
                        preds.append(0)
                    if last_pos_source_id in pos_imgs or last_pos_source_id in pos_texts:
                        labels.append(1)
                        query2result[qid]["true"] = query2result[qid].get("true", []) + [last_pos_source_id]
                    else:
                        labels.append(0)

                for i in range(2, len(D_1[0])):
                    score = D_1[0][i]
                    preds.append(0)
                    scores.append(score)
                    if all_idx[I_1[0][i]] in pos_imgs or all_idx[I[0][i]] in pos_texts:
                        labels.append(1)
                        query2result[qid]["true"] = query2result[qid].get("true", []) + [all_idx[I[0][i]]]
                    else:
                        labels.append(0)
            else:
                for i in range(1, len(D[0])):
                    score = D[0][i]
                    preds.append(0)
                    scores.append(score)
                    if all_idx[I[0][i]] in pos_imgs or all_idx[I[0][i]] in pos_texts:
                        labels.append(1)
                        query2result[qid]["true"] = query2result[qid].get("true", []) + [all_idx[I[0][i]]]
                    else:
                        labels.append(0)



thresholds = np.linspace(0, 1, 100)
f1_scores = []
for threshold in thresholds:
    predictions = [1 if score>= threshold else 0 for score in scores]
    f1_scores.append(f1_score(labels, predictions))
threshold = thresholds[np.argmax(np.array(f1_scores))]
print("optimal threshold is", threshold)

final_score = f1_score(labels, preds)
precision = precision_score(labels, preds)

recall = recall_score(labels, preds)
print(f"final F1 score is {final_score}, precision {precision}, recall {recall}")
# final F1 score is 0.669527896995708
#
with open("/Users/baonguyen/Desktop/MMML Project Code/UniVL-DR/UniVL-DR/exp_1/retrieval_1.json", "w") as outfile:
    json.dump(query2result, outfile)









