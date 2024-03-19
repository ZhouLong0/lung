from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
from scipy.stats import mode
from src.fewshot.utils import Logger, extract_features, get_one_hot
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


class SIMPLE_SHOT(object):
    def __init__(self, model, device, log_file, args):
        self.device = device
        self.norm_type = args.norm_type
        self.n_ways = args.n_ways
        self.num_NN = args.num_NN
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []

    def record_info(self, y_q, logits_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        list_classes = ['AM', 'AN', 'NT', 'RE', 'VE']
        a = len(list_classes)
        u = logits_q.softmax(2).detach()
        preds_q = logits_q.argmax(2)
        conf = u.max(2).values

        # if the file average_confusion_matrix.npy does not exist, create it, otherwise load it
        cf_matrix = np.zeros((1, a, a))
        predictions, groundtruth, confidences = [], [], []
        for i in range(len(preds_q)):
            cf_matrix[0, :, :] += confusion_matrix(y_q[i].cpu().numpy().flatten(), preds_q[i].cpu().numpy().flatten(), labels=list(range(a)))
            predictions += list(preds_q[i].cpu().numpy())
            groundtruth += list(y_q[i].cpu().numpy())
            confidences += list(conf[i].cpu().numpy())
        cf_matrix = cf_matrix / len(preds_q)
        try:
            cf_matrix_old = np.load('confusion_matrix/average_confusion_matrix.npy')
            np.save('confusion_matrix/average_confusion_matrix.npy', np.concatenate((cf_matrix_old, cf_matrix), axis=0))
        except:
            np.save('confusion_matrix/average_confusion_matrix.npy', cf_matrix)

        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

        return groundtruth, predictions, confidences
    

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps,
                'acc': self.test_acc}
    
    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits =  (samples.matmul(self.w.transpose(1, 2)) \
                              - 1 / 2 * (self.w**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits
    

    def normalization(self, z_s, z_q, train_mean):
        """
            inputs:
                z_s : np.Array of shape [n_task, s_shot, feature_dim]
                z_q : np.Array of shape [n_task, q_shot, feature_dim]
                train_mean: np.Array of shape [feature_dim]
        """
        z_s = z_s.cpu()
        z_q = z_q.cpu()
        # CL2N Normalization
        if self.norm_type == 'CL2N':
            z_s = z_s - train_mean
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q - train_mean
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        # L2 Normalization
        elif self.norm_type == 'L2N':
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        return z_s, z_q

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        if self.norm_type == 'CL2N':
            train_mean = task_dic['train_mean']
        else:
            train_mean = None

        # Perform normalizations required
        #support, query = self.normalization(z_s=support, z_q=query, train_mean=train_mean)

        
        # Transfer tensors to GPU if needed
        support = support.to(self.device).float() #.double()
        query = query.to(self.device).float() #.double()
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic
        print("support", support.shape)
        print("query", query.shape)

        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)

        # Run adaptation
        truth, prediction, confidence = self.run_prediction(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs, truth, prediction, confidence

    def run_prediction(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the Simple inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        records :
            accuracy
            inference time
        """
        t0 = time.time()
        time_list = []
        self.logger.info(" ==> Executing predictions on {} shot tasks...".format(shot))
        out_list = []

        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s).float()
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.w = weights / counts

        logits_q = self.get_logits(query).detach()

        truth, prediction, confidence = self.record_info(y_q=y_q, logits_q=logits_q)  
        print("pred",prediction)
        print("truth",truth)
        return truth, prediction, confidence
