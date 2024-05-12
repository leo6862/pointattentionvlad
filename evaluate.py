import argparse
import math
import numpy as np
import socket
import time
import importlib
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
from torch.backends import cudnn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import *
import models.pointattentionvlad as PointAttentionVLAD
from tensorboardX import SummaryWriter
import loss.pointnetvlad_loss

import config as cfg

cudnn.enabled = True


def evaluate():
    model = PointAttentionVLAD.PointAttentionVLAD()
    model = model.to(device)

    resume_filename = cfg.LOG_DIR + "model.ckpt"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    print(evaluate_model(model))


def evaluate_model(model):
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall_yyj(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS,DATABASE_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    return ave_one_percent_recall


def evaluate_model_kaist(model):
    # print("cfg.EVAL_DATABASE_FILE = {}".format(cfg.EVAL_DATABASE_FILE))
    with open(cfg.EVAL_DATABASE_FILE, 'rb') as handle:
        testing_pkl = pickle.load(handle)
        print("Queries Loaded.")
        features = []
        for i in range(len(testing_pkl.keys())//cfg.EVAL_SIZE):
            file_idxs = range(i * cfg.EVAL_SIZE , (i+1) * cfg.EVAL_SIZE)
            file_names = []
            for idx in file_idxs:
                file_names.append(testing_pkl[idx]["query"])
            queries = load_processed_pc_files(file_names)

            with torch.no_grad():
                feed_tensor = torch.from_numpy(queries).float()
                feed_tensor = feed_tensor.unsqueeze(1)
                feed_tensor = feed_tensor.to(device)
                out = model(feed_tensor)    
            out = out.detach().cpu().numpy()
            out = np.squeeze(out)
            # print("out[:].shape = {}".format(out.shape)) #(40,256)
            for feature_ind in range(len(out)):
                features.append(out[feature_ind])
            
        # Handle ege case 
        index_edge = len(testing_pkl) // cfg.EVAL_SIZE * cfg.EVAL_SIZE
        if index_edge < len(testing_pkl.keys()):
            file_names = []
            for edge_ind in range(index_edge,len(testing_pkl.keys())):
                file_names.append(testing_pkl[edge_ind]["query"])
            queries = load_processed_pc_files(file_names)

            with torch.no_grad():
                feed_tensor = torch.from_numpy(queries).float()
                feed_tensor = feed_tensor.unsqueeze(1)
                feed_tensor = feed_tensor.to(device)
                print(feed_tensor.shape)
                o1 = model(feed_tensor)

            output = o1.detach().cpu().numpy()
            output = np.squeeze(output)
            for feature_ind in range(len(output)):
                features.append(output[feature_ind])
        features_ = np.vstack(features)

        #knn search 
        kd_tree = KDTree(features_)
        recall_at_n = [0 for _ in range(26)]
        top_1_success_num = 0
        top_1_percent_success_num = 0
        one_percent_thre = int(len(testing_pkl.keys()) / 100)
        for ind,data_dict in testing_pkl.items():
            current_feature = np.array(features[ind]).reshape(1,-1)
            positives = data_dict["positives"]
            distances, indices = kd_tree.query(
            current_feature,k=30)
            indices = np.squeeze(indices)
            if indices[1] in positives:
                top_1_success_num += 1
            indices = indices.tolist()
            if len(list(set(indices[0:one_percent_thre]).intersection(set(positives)))) > 0:
                top_1_percent_success_num += 1
            for n_ind in range(1,len(recall_at_n)):
                if indices[n_ind] in positives:
                    for ind in range(n_ind,len(recall_at_n)):
                        recall_at_n[ind] += 1
                    break
        
        one_percent_recall = (top_1_percent_success_num/float(len(testing_pkl.items())))*100
        top_one_recall = (top_1_success_num/float(len(testing_pkl.items())))*100
        recall_at_n = [recall / float(len(testing_pkl.items()))*100 for recall in recall_at_n]
        
        print(recall_at_n)
        #输出结果
        with open(cfg.OUTPUT_FILE, "w") as output:
            output.write("Average Recall @N:\n")
            output.write(str(recall_at_n))
            output.write("\n\n")
            # output.write(str(average_similarity))
            # output.write("\n\n")
            output.write("Average Top 1% Recall:\n")
            output.write(str(one_percent_recall))
            output.write("Average Top 1 Recall:\n")
            output.write(str(top_one_recall))
    return one_percent_recall
            



def evaluate_model_nn_neighbor(model):
    DATABASE_SETS = get_sets_dict(cfg.EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(cfg.EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []

    for i in range(len(DATABASE_SETS)):
        DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))

    for j in range(len(QUERY_SETS)):
        QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))

    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if (m == n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(
                m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print()
    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(cfg.OUTPUT_FILE, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))

    return ave_one_percent_recall




def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.EVAL_BATCH_SIZE * \
        (1 + cfg.EVAL_POSITIVES_PER_QUERY + cfg.EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_processed_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_processed_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output

def get_recall_yyj(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS,DATABASE_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        # print('true_neighbors = {}'.format(true_neighbors))
        # print('QUERY_SETS[n][i] = {}'.format(QUERY_SETS[n][i]))
        # time.sleep(1)
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        # print('indices[0] = {}'.format(indices[0]))
        # print('QUERY_SETS[m] = {}'.format(QUERY_SETS[m]))
        res = indices[0].tolist()
        # for kk in range(len(res)):
        #     print(' top {} candidates  =  \n {}'.format(kk,DATABASE_SETS[m][indices[0][kk]]))
        #     if kk== 4:
        #         break
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        print('true_neighbors = {}'.format(true_neighbors))
        print('QUERY_SETS[n][i] = {}'.format(QUERY_SETS[n][i]))
        time.sleep(1)
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break
        res = indices[0].tolist()
        for kk in range(len(res)):
            print(' top {} candidates  =  \n {}'.format(kk,QUERY_SETS[m][indices[0][kk]]))
            if kk== 4:
                break
        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


if __name__ == "__main__":
    # params
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='../../dataset/',
                        help='PointNetVlad Dataset Folder')
    FLAGS = parser.parse_args()

    #BATCH_SIZE = FLAGS.batch_size
    #cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.NUM_POINTS = 4096
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate

    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.EVAL_DATABASE_FILE = 'generating_queries/oxford_evaluation_database.pickle'
    cfg.EVAL_QUERY_FILE = 'generating_queries/oxford_evaluation_query.pickle'

    cfg.LOG_DIR = 'log/'
    cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results.txt'
    cfg.MODEL_FILENAME = "model.ckpt"

    cfg.DATASET_FOLDER = FLAGS.dataset_folder
   
    evaluate()
