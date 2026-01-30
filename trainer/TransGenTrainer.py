# -*- coding: utf-8 -*-
# @Author: Xiaoning Qi
# @Date:   2022-06-23 02:24:40
# @Last Modified by:   Xiaoning Qi
# @Last Modified time: 2024-10-31 15:26:11
import os
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error


from data.Dataset import  DrugDoseAnnDataset
from models.PRnet import PRnet
from models.TransGen.model import TranSiGen

from ._utils import train_valid_test



    
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

class TransGenTrainer:
    """
    This class contains the implementation of the PRnetTrainer Trainer
    Parameters
    ----------
    model: PRnet
    adata: : `~anndata.AnnData`
        Annotated Data Matrix for training PRnet.
    batch_size: integer
        size of each batch to be fed to network.
    comb_num: int
        Number of combined compounds.
    shuffle: bool
        if `True` shuffles the training dataset.
    split_key: string
        Attributes of data split.
    model_save_dir: string
        Save dir of model. 
    x_dimension: int
        Dimention of x
    hidden_layer_sizes: list
        A list of hidden layer sizes
    z_dimension: int
        Dimention of latent space
    adaptor_layer_sizes: list
        A list of adaptor layer sizes
    comb_dimension: int
        Dimention of perturbation latent space
    drug_dimension: int
        Dimention of rFCGP
    n_genes: int
        Dimention of different expressed gene
    n_epochs: int
        Number of epochs to iterate and optimize network weights.
    train_frac: Float
        Defines the fraction of data that is used for training and data that is used for validation.
    dr_rate: float
        dropout_rate
    loss: list
        Loss of model, subset of 'NB', 'GUSS', 'KL', 'MSE'
    obs_key:
        observation key of data
    """
    def __init__(self, adata, batch_size = 32, comb_num = 2, shuffle = True, split_key='random_split', model_save_dir = './checkpoint/',results_save_dir = './results/', x_dimension = 5000, hidden_layer_sizes = [128], z_dimension = 64, adaptor_layer_sizes = [128], comb_dimension = 64, drug_dimension = 1031, n_genes=20,  dr_rate = 0.05, loss = ['GUSS'], obs_key = 'cov_drug_name', **kwargs): # maybe add more parameters
        
        assert set(loss).issubset(['NB', 'GUSS', 'KL', 'MSE']), "loss should be subset of ['NB', 'GUSS', 'KL', 'MSE']"

        self.x_dim = x_dimension
        self.split_key = split_key
        self.z_dimension = z_dimension
        self.comb_dimension = comb_dimension
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        local_out = './checkpoint/TransGen'#???
        isExists = os.path.exists(local_out)
        if not isExists:
            os.makedirs(local_out)
            print('Directory created successfully')
        else:
            print('Directory already exists')
        #self.model = PRnet(adata, x_dimension=self.x_dim, hidden_layer_sizes=hidden_layer_sizes, z_dimension=z_dimension, adaptor_layer_sizes=adaptor_layer_sizes, comb_dimension=comb_dimension, comb_num=comb_num, drug_dimension=drug_dimension,dr_rate=dr_rate)
        self.model=TranSiGen(n_genes=978, n_latent=z_dimension, n_en_hidden=[1200], n_de_hidden=[800],
                            features_dim=2304, features_embed_dim=[400],
                            init_w=True, beta=0.1, device=self.device, dropout=0.1,
                            path_model=local_out, random_seed=364039)
        self.model.to(self.device)
        self.model_save_dir = model_save_dir
        self.results_save_dir = results_save_dir
        self.loss = loss
        #self.modelPGM = self.model.get_PGM()


        self.seed = kwargs.get("seed", 2024)
        torch.manual_seed(self.seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(self.seed)
        #     if(torch.cuda.device_count() > 1):
        #         self.modelPGM = nn.DataParallel(self.modelPGM, device_ids=[i for i in range(torch.cuda.device_count())])
            

        #self.modelPGM = self.modelPGM.to(self.device)

        #self.modelPGM.apply(self.weight_init)
        #print(self.modelPGM)


        self.adata = adata
        #self.adata_deg_list = adata.uns['rank_genes_groups_cov']
        self.de_n_genes = n_genes
        self.adata_var_names = adata.var_names
        self.train_data, self.valid_data, self.test_data = train_valid_test(self.adata, split_key = split_key)

        
        if self.train_data is not None:
            self.train_dataset = DrugDoseAnnDataset(self.train_data, dtype='train', obs_key=obs_key, comb_num=comb_num)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,num_workers=12,pin_memory=True)
        if self.valid_data is not None:
            self.valid_dataset = DrugDoseAnnDataset(self.valid_data, dtype='valid', obs_key=obs_key, comb_num=comb_num)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12,pin_memory=True)
        if self.test_data is not None:
            self.test_dataset = DrugDoseAnnDataset(self.test_data, dtype='test', obs_key=obs_key, comb_num=comb_num)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=12,pin_memory=True)

        if set(['NB']).issubset(loss):
            self.criterion = NBLoss()
        if set(['GUSS']).issubset(loss):
            self.criterion = nn.GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        self.shuffle = shuffle
        self.batch_size = batch_size

        # Optimization attributes

        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.best_state_dictPGM = None


        self.PGM_losses = []
        self.r2_score_mean = []
        self.r2_score_var = []
        self.mse_score = []
        self.r2_score_mean_de = []
        self.r2_score_var_de = []
        self.mse_score_de = []
        self.best_mse = np.inf
        self.patient = 0











    def train(self, n_epochs = 100, lr = 0.001, weight_decay= 1e-8, scheduler_factor=0.5,scheduler_patience=10,**extras_kwargs):
        self.n_epochs = n_epochs
        self.epoch=0
        self.params = filter(lambda p: p.requires_grad, self.model.parameters())
        # paramsPGM = filter(lambda p: p.requires_grad, self.modelPGM.parameters())

        self.optimPGM = torch.optim.Adam(
            self.params, lr=lr, weight_decay= weight_decay) # consider changing the param. like weight_decay, eps, etc.
        #self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(self.optimPGM, step_size=10)
        self.scheduler_autoencoder = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimPGM, 'min',factor=scheduler_factor,verbose=1,min_lr=1e-8,patience=scheduler_patience)
        train_size = 0
        loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert', 'kld_x1', 'kld_x2', 'kld_pert']
        best_value = np.inf
        best_epoch = 0
        for self.epoch in range(self.n_epochs):
            loop = tqdm(enumerate(self.train_dataloader), total =len(self.train_dataloader))
            train_size = 0
            loss_value = 0
            train_dict = defaultdict(float)
            for i, data in loop:
                self.model.zero_grad()
                (control, target) = data['features']
                encode_label = data['label']

                # 将数据显式移动到 GPU
                control = control.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.float32)
                encode_label = encode_label.to(self.device, dtype=torch.float32)

                # 检查数据是否在 GPU 上
                assert control.device == self.device, f"Control is on {control.device}, expected {self.device}"
                assert target.device == self.device, f"Target is on {target.device}, expected {self.device}"
                assert encode_label.device == self.device, f"Encode label is on {encode_label.device}, expected {self.device}"

                # 检查模型参数是否在 GPU 上
                for param in self.model.parameters():
                    assert param.device == self.device, f"Model parameter is on {param.device}, expected {self.device}"

                # 正常训练逻辑继续...
                if control.shape[0] == 1:
                    continue
                train_size += control.shape[0]
                self.optimPGM.zero_grad()

                x1_rec, mu1, logvar1, x2_pert, mu_pred, logvar_pred, z2_pred = self.model.forward(control, encode_label)
                z2, mu2, logvar2 = self.model.encode_x2(target)
                x2_rec = self.model.decode_x2(z2)
                loss, _, _, _, _, _, _ = self.model.loss(control, x1_rec, mu1, logvar1, target, x2_rec, mu2, logvar2,
                                                   x2_pert, mu_pred, logvar_pred)
                loss_ls=self.model.loss(control, x1_rec, mu1, logvar1, target, x2_rec, mu2, logvar2,
                                                   x2_pert, mu_pred, logvar_pred)
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        train_dict[k] += loss_ls[idx].item()




                # x1_array = control.clone()
                # x1_rec_array = x1_rec.clone()
                # x2_array = target.clone()
                # x2_rec_array = x2_rec.clone()
                # x2_pred_array = x2_pert.clone()
                loss_value += loss.item()
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  # 进行梯度裁剪
                self.optimPGM.step()
            for k in train_dict.keys():
                train_dict[k] = train_dict[k] / train_size
            train_loss = train_dict['loss']
            train_mse_x1 = train_dict['mse_x1']
            train_mse_x2 = train_dict['mse_x2']
            train_mse_pert = train_dict['mse_pert']
            train_kld_x1 = train_dict['kld_x1']
            train_kld_x2 = train_dict['kld_x2']
            train_kld_pert = train_dict['kld_pert']
            loop_v = tqdm(enumerate(self.valid_dataloader), total =len(self.valid_dataloader))
            self.r2_sum_mean = 0
            self.r2_sum_var = 0
            self.mse_sum = 0
            self.r2_sum_mean_de = 0
            self.r2_sum_var_de = 0
            self.mse_sum_de = 0
            test_dict = defaultdict(float)
            metrics_dict_all = defaultdict(float)
            metrics_dict_all_ls = defaultdict(list)
            test_size = 0
            metrics_func=None
            for j, vdata in loop_v:
                (control, target) = vdata['features']
                encode_label = vdata['label']
                data_cov_drug = vdata['cov_drug']
                target = target.to(self.device, dtype=torch.float32)
                encode_label = encode_label.to(self.device, dtype=torch.float32)
                control = control.to(self.device, dtype=torch.float32)
                test_size += target.shape[0]

                x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.model.forward(control, encode_label)
                z2, mu2, logvar2 = self.model.encode_x2(target)
                x2_rec = self.model.decode_x2(z2)
                loss_ls = self.model.loss(control, x1_rec, mu1, logvar1, target, x2_rec, mu2, logvar2, x2_pred, mu_pred,
                                    logvar_pred)
                if loss_item != None:
                    for idx, k in enumerate(loss_item):
                        test_dict[k] += loss_ls[idx].item()

                if metrics_func != None:
                    metrics_dict, metrics_dict_ls = self.eval_x_reconstruction(control, x1_rec, target, x2_rec,
                                                                               x2_pred, metrics_func=metrics_func)
                    for k in metrics_dict.keys():
                        metrics_dict_all[k] += metrics_dict[k]
                    for k in metrics_dict_ls.keys():
                        metrics_dict_all_ls[k] += metrics_dict_ls[k]

                try:
                    x1_array = torch.cat([x1_array, control], dim=0)
                    x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                    x2_array = torch.cat([x2_array, target], dim=0)
                    x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                    x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)


                except:
                    x1_array = control.clone()
                    x1_rec_array = x1_rec.clone()
                    x2_array = target.clone()
                    x2_rec_array = x2_rec.clone()
                    x2_pred_array = x2_pred.clone()

            for k in test_dict.keys():
                test_dict[k] = test_dict[k] / test_size



            test_loss = test_dict['loss']
            test_mse_x1 = test_dict['mse_x1']
            test_mse_x2 = test_dict['mse_x2']
            test_mse_pert = test_dict['mse_pert']
            test_kld_x1 = test_dict['kld_x1']
            test_kld_x2 = test_dict['kld_x2']
            test_kld_pert = test_dict['kld_pert']
            print(
                '[Epoch %d] | loss: %.3f, mse_x1_rec: %.3f, mse_x2_rec: %.3f, mse_pert: %.3f, kld_x1: %.3f, kld_x2: %.3f, kld_pert: %.3f| '
                'valid_loss: %.3f, valid_mse_x1_rec: %.3f, valid_mse_x2_rec: %.3f, valid_mse_pert: %.3f, valid_kld_x1: %.3f, valid_kld_x2: %.3f, valid_kld_pert: %.3f|'
                % (self.epoch, train_loss, train_mse_x1, train_mse_x2, train_mse_pert, train_kld_x1, train_kld_x2,
                   train_kld_pert,
                   test_loss, test_mse_x1, test_mse_x2, test_mse_pert, test_kld_x1, test_kld_x2, test_kld_pert),
                flush=True)

            if test_loss < best_value:
                best_value = test_loss


                #torch.save(self, self.path_model + 'best_model.pt')
                self.best_state_dictG = self.model.state_dict()
                torch.save(self.best_state_dictG, self.model_save_dir + self.split_key + '_TranSiGen_best_epoch_all.pt')

         
    def eval(self):
        self.model.load_state_dict(torch.load("./drug_split_4_best_epoch_all.pt"))
        self.model.eval()
        x_true_array = np.zeros((0, self.x_dim))
        y_true_array = np.zeros((0, self.x_dim))
        y_pre_array = np.zeros((0, self.x_dim))
        cov_drug_list = []
        loop_v = tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader))

        loss_item = ['loss', 'mse_x1', 'mse_x2', 'mse_pert', 'kld_x1', 'kld_x2', 'kld_pert']
        test_dict = defaultdict(float)
        metrics_dict_all = defaultdict(float)
        metrics_dict_all_ls = defaultdict(list)
        test_size = 0

        metrics_func = ['pearson', 'rmse',
                        'precision100']
        for j, vdata in loop_v:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']

            cov_drug_list = cov_drug_list + data_cov_drug
            import os

            # 确保保存结果的文件夹存在
            save_folder = self.results_save_dir
            os.makedirs(save_folder, exist_ok=True)

            with open(self.results_save_dir+self.split_key+"_cov_drug_array_TranSiGen.csv", 'a') as f:
                for i in data_cov_drug:
                    f.write(i+'\n')
            target = target.to(self.device, dtype=torch.float32)
            encode_label = encode_label.to(self.device, dtype=torch.float32)#1024 1024
            control = control.to(self.device, dtype=torch.float32)
            test_size += target.shape[0]

            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = self.model.forward(control, encode_label)
            z2, mu2, logvar2 = self.model.encode_x2(target)
            x2_rec = self.model.decode_x2(z2)
            #-------
            y_pre_array = np.concatenate((y_pre_array, x2_pred.detach().cpu().numpy()), axis=0)

            x_true = control.cpu().numpy()
            x_true_array = np.concatenate((x_true_array, x_true), axis=0)

            y_true = target.cpu().numpy()
            y_true_array = np.concatenate((y_true_array, y_true), axis=0)
            import os

            # 确保保存结果的文件夹存在
            save_folder = self.results_save_dir
            os.makedirs(save_folder, exist_ok=True)
            with open(self.results_save_dir + self.split_key + "_y_true_array_TranSiGen.csv", 'a+') as f:
                np.savetxt(f, y_true, delimiter=",")


            with open(self.results_save_dir+self.split_key+"_y_pre_array_TranSiGen.csv", 'a+') as f:
                np.savetxt(f, x2_pred.detach().cpu().numpy(), delimiter=",")

            with open(self.results_save_dir+self.split_key+"_x_true_array_TranSiGen.csv", 'a+') as f:
                np.savetxt(f, x_true, delimiter=",")
            #----

            #--------------**
            loss_ls = self.model.loss(control, x1_rec, mu1, logvar1, target, x2_rec, mu2, logvar2, x2_pred, mu_pred,
                                      logvar_pred)
            if loss_item != None:
                for idx, k in enumerate(loss_item):
                    test_dict[k] += loss_ls[idx].item()

            if metrics_func != None:
                metrics_dict, metrics_dict_ls = self.model.eval_x_reconstruction(control, x1_rec, target, x2_rec,
                                                                           x2_pred, metrics_func=metrics_func)
                for k in metrics_dict.keys():
                    metrics_dict_all[k] += metrics_dict[k]
                for k in metrics_dict_ls.keys():
                    metrics_dict_all_ls[k] += metrics_dict_ls[k]

            try:
                x1_array = torch.cat([x1_array, control], dim=0)
                x1_rec_array = torch.cat([x1_rec_array, x1_rec], dim=0)
                x2_array = torch.cat([x2_array, target], dim=0)
                x2_rec_array = torch.cat([x2_rec_array, x2_rec], dim=0)
                x2_pred_array = torch.cat([x2_pred_array, x2_pred], dim=0)


            except:
                x1_array = control.clone()
                x1_rec_array = x1_rec.clone()
                x2_array = target.clone()
                x2_rec_array = x2_rec.clone()
                x2_pred_array = x2_pred.clone()
            #------------------------**

        for k in test_dict.keys():
            test_dict[k] = test_dict[k] / test_size
        for k in metrics_dict_all.keys():
            metrics_dict_all[k] = metrics_dict_all[k] / test_size
        test_loss = test_dict['loss']
        test_mse_x1 = test_dict['mse_x1']
        test_mse_x2 = test_dict['mse_x2']
        test_mse_pert = test_dict['mse_pert']
        test_kld_x1 = test_dict['kld_x1']
        test_kld_x2 = test_dict['kld_x2']
        test_kld_pert = test_dict['kld_pert']
        print(
            'valid_loss: %.3f, valid_mse_x1_rec: %.3f, valid_mse_x2_rec: %.3f, valid_mse_pert: %.3f, valid_kld_x1: %.3f, valid_kld_x2: %.3f, valid_kld_pert: %.3f|'
            % (
               test_loss, test_mse_x1, test_mse_x2, test_mse_pert, test_kld_x1, test_kld_x2, test_kld_pert),
            flush=True)
        import pickle

        # 保存字典
        with open('TransGen_metrics_dict_all.pkl', 'wb') as f:
            pickle.dump(metrics_dict_all, f)

        with open('TransGen_metrics_dict_all_ls.pkl', 'wb') as f:
            pickle.dump(metrics_dict_all_ls, f)

        

       

    
 
    @staticmethod
    def _anndataToTensor(adata: AnnData) -> torch.Tensor:
        data_ndarray = adata.X.A
        data_tensor = torch.from_numpy(data_ndarray)
        return data_tensor
    
    def make_noise(self, batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(self.device, dtype=torch.float32)
        return noise

    def weight_init(self, m):  
        # initialize the weights of the model
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(1, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
    

    def get_per_latent(self, model_path):
    
        self.modelPGM.load_state_dict(torch.load(model_path))
        self.modelPGM.eval()

        cov_drug_list = []
        
        latent_array = np.zeros((0, self.comb_dimension))

        loop_t = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
        
        for j, vdata in loop_t:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']
            cov_drug_list = cov_drug_list + data_cov_drug

            control = control.to(self.device, dtype=torch.float32)
            if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
            target = target.to(self.device, dtype=torch.float32)

            encode_label = encode_label.to(self.device, dtype=torch.float32)
           
            b_size = control.size(0)
            noise = self.make_noise(b_size, 10)

            latent = self.model.get_per_latent(control, encode_label, noise)

            latent_array = np.concatenate((latent_array, latent),axis=0)

        return latent_array, cov_drug_list


    def get_latent(self, model_path):
        
        self.modelPGM.load_state_dict(torch.load(model_path))
        self.modelPGM.eval()

        cov_drug_list = []
        
        latent_array = np.zeros((0, self.z_dimension))

        loop_t = tqdm(enumerate(self.test_dataloader), total =len(self.test_dataloader))
        
        for j, vdata in loop_t:
            (control, target) = vdata['features']
            encode_label = vdata['label']
            data_cov_drug = vdata['cov_drug']
            cov_drug_list = cov_drug_list + data_cov_drug

            control = control.to(self.device, dtype=torch.float32)
            if set(['NB']).issubset(self.loss):
                    control = torch.log1p(control)
            target = target.to(self.device, dtype=torch.float32)

            encode_label = encode_label.to(self.device, dtype=torch.float32)
           
            b_size = control.size(0)
            noise = self.make_noise(b_size, 10)
            

            latent = self.model.get_latent(control, encode_label, noise)

            latent_array = np.concatenate((latent_array, latent),axis=0)

        return latent_array, cov_drug_list

    @staticmethod
    def pearson_mean(data1, data2):
        sum_pearson_1 = 0
        sum_pearson_2 = 0
        for i in range(data1.shape[0]):
            pearsonr_ = pearsonr(data1[i], data2[i])
            sum_pearson_1 += pearsonr_[0]
            sum_pearson_2 += pearsonr_[1]
        return sum_pearson_1/data1.shape[0], sum_pearson_2/data1.shape[0]
    
    @staticmethod
    def r2_mean(data1, data2):
        sum_r2_1 = 0
        for i in range(data1.shape[0]):
            r2_score_ = r2_score(data1[i], data2[i])
            sum_r2_1 += r2_score_           
        return sum_r2_1/data1.shape[0]

    @staticmethod
    def _convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
        r"""NB parameterizations conversion. Reference: https://github.com/theislab/chemCPA/tree/main.
    Parameters
    ----------
    mu :
        mean of the NB distribution.
    theta :
        inverse overdispersion.
    eps :
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
        assert (mu is None) == (theta is None), "If using the mu/theta NB parameterization, both parameters must be specified"
        logits = (mu + eps).log() - (theta + eps).log()
        total_count = theta
        return total_count, logits
    
    @staticmethod
    def _sample_z(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed by the Encoder.

        Parameters
            ----------
        mean:
        Mean of the latent Gaussian
            log_var:
        Standard deviation of the latent Gaussian
            Returns
            -------
        Returns Torch Tensor containing latent space encoding of 'x'.
        The computed Tensor of samples with shape [size, z_dim].
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps


class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3). Reference: https://github.com/theislab/chemCPA/tree/main.
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y + 1.0)
            - torch.lgamma(y + theta + eps)
        )
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
            y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)

    @staticmethod
    def _sample_z(mu, log_var):
        
        std = np.exp(0.5 * log_var)
        eps = torch.random.randn(std)
        return mu + std * eps

