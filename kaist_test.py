import argparse
import importlib
import math
import os
import socket
import sys
import torch
import numpy as np
from sklearn.neighbors import KDTree, NearestNeighbors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device = {}'.format(device))

import config as cfg
import evaluate
import loss.pointnetvlad_loss as PNV_loss

import models.pointattentionvlad as PointAttentionVLAD

import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=30,
                    help='Epoch to run [default: 30]')
parser.add_argument('--batch_num_queries', type=int, default=2,
                    help='Batch Size during training [default: 10]')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='Initial learning rate [default: 0.000005]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5,
                    help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2,
                    help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--loss_function', default='quadruplet', choices=[
                    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_false',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_false',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='/home/yyj/dl_dataset/kaist_v3/',
                    help='PointAttentionVLAD Dataset Folder')

FLAGS = parser.parse_args()
cfg.BATCH_NUM_QUERIES = FLAGS.batch_num_queries
#cfg.EVAL_BATCH_SIZE = 12
cfg.NUM_POINTS = 10000
cfg.TRAIN_POSITIVES_PER_QUERY = FLAGS.positives_per_query
cfg.TRAIN_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2
cfg.FEATURE_OUTPUT_DIM = 256

cfg.LOSS_FUNCTION = FLAGS.loss_function
cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch

# cfg.TRAIN_FILE = '/root/autodl-tmp/kaist_v3/training_tuple_v3.pickle'
# cfg.TEST_FILE = '/root/autodl-tmp/kaist_v3/testing_tuple.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

cfg.RESULTS_FOLDER = FLAGS.results_dir

cfg.DATASET_FOLDER = FLAGS.dataset_folder

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)

cfg.BN_INIT_DECAY = 0.5
cfg.BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(cfg.DECAY_STEP)
cfg.BN_DECAY_CLIP = 0.99

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []

TOTAL_ITERATIONS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

#

def evaluate_model_kaist_single_seq(model):
    """
        1. 读取 testing pkl
        2. 将每个testing pkl中的文件读取出来
        3. 
    """
    one_percent_recalls = []
    top_one_recalls = []
    data_root = "/home/yyj/dl_dataset/kaist_v2/"
    test_tuples_single_seqs = ["urban00_testing_tuple.pickle","urban04_testing_tuple.pickle",
                              "urban09_testing_tuple.pickle","urban14_testing_tuple.pickle"]
    all_data_size = 0
    for single_seq in test_tuples_single_seqs:
        pkl_file_path = data_root + single_seq
        print("pkl_file_path = {}".format(pkl_file_path))
        with open(pkl_file_path, 'rb') as handle:
            testing_pkl = pickle.load(handle)
            print("Queries Loaded.")
            features = []
            all_data_size += len(testing_pkl.keys())
            for i in range(len(testing_pkl.keys())//cfg.YYJ_EVAL_SIZE):
                file_idxs = range(i * cfg.YYJ_EVAL_SIZE , (i+1) * cfg.YYJ_EVAL_SIZE)
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
                
            #Handle ege case 
            index_edge = len(testing_pkl) // cfg.YYJ_EVAL_SIZE * cfg.YYJ_EVAL_SIZE
            if index_edge < len(testing_pkl.keys()):
                file_names = []
                for edge_ind in range(index_edge,len(testing_pkl.keys())):
                    file_names.append(testing_pkl[edge_ind]["query"])
                queries = load_processed_pc_files(file_names)

                with torch.no_grad():
                    feed_tensor = torch.from_numpy(queries).float()
                    feed_tensor = feed_tensor.unsqueeze(1)
                    feed_tensor = feed_tensor.to(device)
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
                # print(indices.shape)
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
            
        one_percent_recall = (top_1_percent_success_num/len(testing_pkl.keys()))*100
        top_one_recall = (top_1_success_num/len(testing_pkl.keys()))*100
        recall_at_n = [recall / len(testing_pkl.keys())*100 for recall in recall_at_n]
        one_percent_recalls.append(one_percent_recall)
        top_one_recalls.append(top_one_recall)
        
    final_ave_top_one_recall = 0
    for one_recall in top_one_recalls:
        final_ave_top_one_recall += one_recall
    final_ave_top_one_recall /= len(top_one_recalls)
    
    final_ave_top_one_percent_recall = 0
    for one_percent_recall in one_percent_recalls:
        final_ave_top_one_percent_recall += one_percent_recall
    final_ave_top_one_percent_recall /= len(top_one_recalls)
    with open(cfg.OUTPUT_FILE, "w") as output:
        # output.write("Average Recall @N:\n")
        # output.write(str(recall_at_n))
        # output.write("\n\n")
        # output.write(str(average_similarity))
        # output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(final_ave_top_one_percent_recall))
        output.write("Average Top 1 Recall:\n")
        output.write(str(final_ave_top_one_recall))
    return one_percent_recalls




model = PointAttentionVLAD.PointAttentionVLAD()
model = model.to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())

#TODO modify this
resume_filename = cfg.LOG_DIR + "kaist_v3_92_top_1_recall.ckpt"
print("Resuming From ", resume_filename)
checkpoint = torch.load(resume_filename)
saved_state_dict = checkpoint['state_dict']
starting_epoch = checkpoint['epoch']
TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
model.load_state_dict(saved_state_dict)

model = nn.DataParallel(model)
eval_recall = evaluate.evaluate_model_kaist(model)
log_string('EVAL RECALL: %s' % str(eval_recall))

