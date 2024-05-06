import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ={}".format(device))
import evaluate 
import config as cfg
import models.pointattentionvlad as PointAttentionVLAD
from loading_pointclouds import *


resume_filename = "./pretrained_models/96_6779_recall.ckpt"
print("Resuming From ", resume_filename)
checkpoint = torch.load(resume_filename)
saved_state_dict = checkpoint['state_dict']
model = PointAttentionVLAD.PointAttentionVLAD().cuda()

learning_rate = 0.00001
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, learning_rate)

model.load_state_dict(saved_state_dict)
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


cfg.TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
cfg.TEST_FILE = 'generating_queries/test_queries_baseline.pickle'
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)


eval_recall = evaluate.evaluate_model(model)
print('EVAL RECALL: %s' % str(eval_recall))


def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + 2 + 10 + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)
        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)
            print('out.norm = {}'.format(out.norm(dim=1)))
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)
        
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output

