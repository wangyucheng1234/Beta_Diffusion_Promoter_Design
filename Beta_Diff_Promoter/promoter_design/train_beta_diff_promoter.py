import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import time
import tqdm
import tabix
import pyBigWig
import pandas as pd
from matplotlib import pyplot as plt

from selene_sdk.utils import NonStrandSpecific
from selene_sdk.targets import Target

import sys
sys.path.append("../")
sys.path.append("../external/")

from ddsm import *
from sei import *
from selene_utils import *

from torch.nn.functional import logsigmoid

import logging
import argparse

logger = logging.getLogger()


EPS=torch.finfo(torch.float32).eps
MIN=torch.finfo(torch.float32).tiny

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', type = float, default = 10000.0)
    parser.add_argument('--sigmoid_start', type = float, default = 10.0)
    parser.add_argument('--sigmoid_end', type = float, default = -13.0)
    parser.add_argument('--sigmoid_power', type = float, default = 0.2)
    parser.add_argument('--Scale', type = float, default = 0.5)
    parser.add_argument('--Shift', type = float, default = 0.4)
    parser.add_argument('--KLUB_Scale', type = float, default = 0.5)
    parser.add_argument('--KLUB_Shift', type = float, default = 0.4)
    parser.add_argument('--T', type = int, default = 100)
    parser.add_argument('--lossType', type = str, default = "KLUB") #'KLUB', 'KLUB_marginal','KLUB_conditional','KLUB_AS','ELBO','KLUB_marginal_AS','SNR_Weighted_ELBO','KLUB_conditional_AS','L2','L1'
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--normalize_output', action = 'store_true') #Compensate the logit scale variation in different time. 
    parser.add_argument('--no-normalize_output', dest='normalize_output', action = 'store_false')
    parser.set_defaults(normalize_output=True)

    parser.add_argument("--debug", action = 'store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false') 
    parser.set_defaults(train=True)  
    
    parser.add_argument("--test", action = 'store_true')
    parser.add_argument('--no-test', dest='test', action='store_false') 

    parser.set_defaults(test=True)  
    
    args = parser.parse_args()
    print('Arguments:', args)
    return args

class ModelParameters:
    seifeatures_file = '../data/promoter_design/target.sei.names'
    seimodel_file = '../data/promoter_design/best.sei.model.pth.tar'

    ref_file = '../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    ref_file_mmap = '../data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
    tsses_file = '../data/promoter_design/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

    fantom_files = [
                    "../data/promoter_design/agg.plus.bw.bedgraph.bw",
                    "../data/promoter_design/agg.minus.bw.bedgraph.bw"
                    ]
    fantom_blacklist_files = [
         "../data/promoter_design/fantom.blacklist8.plus.bed.gz",
         "../data/promoter_design/fantom.blacklist8.minus.bed.gz"
        ]

    diffusion_weights_file = 'steps400.cat4.speed_balance.time4.0.samples100000.pth'

    device = 'cuda'
    val_batch_size = 1
    num_workers = 4
    ncat = 4
    num_epochs = 200
    
    lr = 5e-4

class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat


class TSSDatasetS(Dataset):
    def __init__(self, config, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
        self.shuffle = False

        self.genome = MemmapGenome(
            input_path=config.ref_file,
            memmapfile=config.ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            config.fantom_files,
            ['cage_plus', 'cage_minus'],
            (2000,),
            config.fantom_blacklist_files
        )

        self.tsses = pd.read_table(config.tsses_file, sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]

        self.chr_lens = self.genome.get_chr_lens()
        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = self.tsses['chr'].values[tssi], self.tsses['TSS'].values[tssi], self.tsses['strand'].values[
            tssi]
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
                                                   pos + int(self.seqlength / 2) + offset, strand)

        signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
                                                pos + int(self.seqlength / 2) + offset)
        if strand == '-':
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

    def reset(self):
        np.random.seed(0)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, embed_dim=256, time_step=0.01):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        n = 256
        self.linear = nn.Conv1d(5, n, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList([nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, padding=4),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)])

        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.scale = nn.Parameter(torch.ones(1))
        self.final = nn.Sequential(nn.Conv1d(n, n, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv1d(n, 4, kernel_size=1))
        # self.register_buffer("time_dependent_weights", time_dependent_weights)
        self.time_step = time_step

    def forward(self, x, t, t_ind=None, return_a=False):
        # Obtain the Gaussian random feature embedding for t
        # embed: [N, embed_dim]
        embed = self.act(self.embed(t / 2))
        # Encoding path
        # x: NLC -> NCL
        out = x.permute(0, 2, 1)
        out = self.act(self.linear(out))

        # pos encoding
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = self.act(block(norm(out + dense(embed)[:, :, None])))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h

        out = self.final(out)

        out = out.permute(0, 2, 1)

        if self.time_dependent_weights is not None:
            t_step = (t / self.time_step) - 1
            w0 = self.time_dependent_weights[t_step.long()]
            w1 = self.time_dependent_weights[torch.clip(t_step + 1, max=len(self.time_dependent_weights) - 1).long()]
            out = out * (w0 + (t_step - t_step.floor()) * (w1 - w0))[:, None, None]
    
        out = out - out.mean(axis=-1, keepdims=True)
        # out = torch.softmax(out, axis = 2)
        return out

class BetaDiffPrecond(torch.nn.Module):
#class BetaDiffusionPrecond(torch.nn.Module):
    def __init__(self,
        # img_resolution,                 # Image resolution.
        # img_channels,                   # Number of color channels.
        # label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        #model_type      = 'SongUNet',   # Class name of the underlying model.
        model_type      = 'None',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        # self.img_resolution = img_resolution
        # self.img_channels = img_channels
        # self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.M = M
        self.epsilon_t = epsilon_t
        # self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)
        self.model = ScoreNet()

    def forward(self, x, logit_alpha, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        logit_alpha = logit_alpha.to(torch.float32).reshape(-1, 1, 1)

        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        #dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        dtype = torch.float32
        c_skip = 1
        c_out = 1
        #c_out = -sigma
        #c_in = 1 / (sigma ** 2 + 1).sqrt()
        
        c_noise = -logit_alpha/8.0
        #F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        # F_x = self.model( x.to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        #Check correctness
        # print("c_noise", c_noise.shape)
        F_x = self.model(torch.cat([x, class_labels], -1), c_noise[:, 0, 0])
        assert F_x.dtype == dtype
        #D_x = c_skip * x + c_out * F_x.to(torch.float32)
        #D_x = F_x.to(torch.float32)
        #D_x = c_skip*x + c_out*F_x.to(torch.float32)
        D_x = x + F_x.to(torch.float32)
        return D_x   

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class BetaDiffLoss:
    def __init__(self, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, Scale=None, Shift=None, KLUB_Scale = None, KLUB_Shift = None, T=200, epsilon_t=1e-5,lossType='KLUB', normalize_output = True):
        self.eta = eta
        self.sigmoid_start = sigmoid_start
        self.sigmoid_end = sigmoid_end
        self.sigmoid_power = sigmoid_power
        self.Scale = Scale
        self.Shift = Shift
        self.KLUB_Scale = KLUB_Scale
        self.KLUB_Shift = KLUB_Shift
        self.T = T
        self.lossType = lossType
        self.epsilon_t = epsilon_t
        self.min = torch.finfo(torch.float32).tiny
        self.eps = torch.finfo(torch.float32).eps        
        self.normalize_output = normalize_output

    def __call__(self, net, images, labels=None):
        if self.T == 0:
            if 1:
                rnd_uniform = torch.rand([images.shape[0], 1, 1], device=images.device)
                rnd_position = 1 + rnd_uniform * (self.epsilon_t - 1)
                #self.epsilon_t + rnd_uniform * (1.0-self.epsilon_t)
                logit_alpha = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
                #rnd_position_previous = (rnd_position - 1/self.T).clamp(min=0)
                rnd_position_previous = rnd_position*0.95
                logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position_previous**self.sigmoid_power)

                alpha = logit_alpha.sigmoid()
                alpha_previous = logit_alpha_previous.sigmoid()

                delta  = (logit_alpha_previous.to(torch.float64).sigmoid()-logit_alpha.to(torch.float64).sigmoid()).to(torch.float32)
            else:
                rnd_position = torch.rand([images.shape[0], 1, 1], device=images.device)
                rnd_position_previous = rnd_position*0.95
                #alpha = (1e-6)**rnd_position
                
                alpha = (torch.tensor(2e-6,device=images.device).log().to(torch.float64)*rnd_position).exp()
                alpha_previous = (torch.tensor(1e-6,device=images.device).log().to(torch.float64)*rnd_position_previous).exp()
                
                delta = (alpha_previous-alpha).to(torch.float32)
                
                
                if 0: 
                    logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
                    alpha_previous = logit_alpha_previous.sigmoid()
                
                    alpha = alpha_previous*0.95
                    delta = alpha_previous*0.05
                
                logit_alpha = alpha.logit().to(torch.float32)
                alpha = alpha.to(torch.float32)
        else:
            # step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
            rnd_position = torch.randint(low = 1, high = T + 1, size = (images.shape[0], 1, 1), device=images.device)/T
            rnd_position_previous = rnd_position - 1/T
            logit_alpha = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
            #rnd_position_previous = (rnd_position - 1/self.T).clamp(min=0)
            logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position_previous**self.sigmoid_power)

            alpha = logit_alpha.sigmoid()
            alpha_previous = logit_alpha_previous.sigmoid()

            delta  = (logit_alpha_previous.to(torch.float64).sigmoid()-logit_alpha.to(torch.float64).sigmoid()).to(torch.float32)            

        eta = torch.ones([images.shape[0], 1, 1], device=images.device) * self.eta

        x0 = images
        x0 = x0.clamp(0,1) * self.Scale + self.Shift

        log_u = self.log_gamma( (self.eta * alpha * x0).to(torch.float32))
        log_v = self.log_gamma( (self.eta - self.eta * alpha * x0).to(torch.float32))

        logit_x_t = (log_u - log_v).to(images.device)        
        #log_alpha = logsigmoid(logit_alpha)
        xmin = self.Shift
        xmax = self.Shift + self.Scale
        xmean = self.Shift+self.Scale/2.0
        E1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).lgamma() - (self.eta * alpha * xmin).lgamma())
        E2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).lgamma() - (self.eta-self.eta * alpha * xmax).lgamma())
        E_logit_x_t =  E1 - E2
        V1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).digamma() - (self.eta * alpha * xmin).digamma())
        V2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).digamma() - (self.eta-self.eta * alpha * xmax).digamma())
        if 1:
            #V3 = (((self.eta * alpha * xmean).digamma())**2- E1**2).clamp(0)
            #V4 = (((self.eta-self.eta * alpha * xmean).digamma())**2- E2**2).clamp(0)
            #V3 = E1**2
            #V4 = E2**2
            grids = (torch.arange(0,101,device=images.device)/100)*self.Scale+self.Shift
            #grids = (torch.arange(0,1001,device=images.device)/1000 +0.5/1000)*self.Scale+self.Shift
            #grids = grids[:-1]
            alpha_x = alpha[:,:,0]*grids.unsqueeze(0)
            #print(alpha_x.shape)
            
            V3 =  ((self.eta * alpha_x).digamma())**2
            V3[:,0] = (V3[:,0]+V3[:,-1])/2
            V3 = V3[:,:-1]
            V3 = (V3.mean(dim=1).unsqueeze(1).unsqueeze(2)- E1**2).clamp(0)  
            
            V4 = ((self.eta - self.eta*alpha_x).digamma())**2
            V4[:,0] = (V4[:,0]+V4[:,-1])/2
            V4 = V4[:,:-1]
            V4 = (V4.mean(dim=1).unsqueeze(1).unsqueeze(2)- E2**2).clamp(0)
            
            #V3 = (( ((self.eta * alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)           
            #V4 = (( ((self.eta - self.eta*alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)

        
        else:
            grids = (torch.arange(0,101,device=images.device)/100 +0.5/100)*self.Scale+self.Shift
            #grids = (torch.arange(0,1001,device=images.device)/1000 +0.5/1000)*self.Scale+self.Shift
            grids = grids[:-1]
            alpha_x = alpha[:,:,0,0]*grids.unsqueeze(0)
            
            V3 = (( ((self.eta * alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)           
            V4 = (( ((self.eta - self.eta*alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)
        
        std_logit_x_t = (V1+V2+V3+V4).sqrt()

        logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha,labels)

        if self.normalize_output:          
            
            variance = (1/4) * (torch.polygamma(1, eta * alpha * (self.Scale + self.Shift)) - torch.polygamma(1, eta * (1- alpha * (self.Scale+self.Shift)))) + \
                    (3/4) * (torch.polygamma(1, eta * alpha * self.Shift) - torch.polygamma(1, eta * (1 - alpha * self.Shift))) + \
                    (3/16) * ((torch.special.digamma(eta * alpha * (self.Scale + self.Shift)) - torch.special.digamma(eta * (1 - alpha * (self.Scale + self.Shift)))) - \
                                (torch.special.digamma(eta * alpha * self.Shift) - torch.special.digamma(eta * (1 - alpha * self.Shift))))**2

            std = torch.clamp(torch.sqrt(variance) * (eta * alpha * self.Scale), max = 20)

            # mean = (1/4) * (torch.special.digamma(eta * alpha * (self.Scale + self.Shift)) - torch.special.digamma(eta * (1- alpha * (self.Scale+self.Shift)))) + \
            #         (3/4) * (torch.special.digamma(eta * alpha * self.Shift) - torch.special.digamma(eta * (1 - alpha * self.Shift)))

            # mean = mean * (eta * alpha * self.Scale) - torch.lgamma(eta * alpha * (self.Scale + self.Shift)) - torch.lgamma(eta * (1 - alpha * (self.Scale + self.Shift))) + \
            #             torch.lgamma(eta * alpha * (self.Shift)) + torch.lgamma(eta * (1 - alpha * (self.Shift))) 

            # mean = torch.clamp(mean, min = -20)

            # logit_x0_hat = logit_x0_hat * std + mean
            logit_x0_hat = logit_x0_hat * std

        # x0_hat = torch.sigmoid(logit_x0_hat)* self.Scale + self.Shift
        # loss = self.compute_loss(x0, x0_hat, alpha, alpha_previous, eta, delta,logit_x_t)

        x0_hat = torch.sigmoid(logit_x0_hat)* self.KLUB_Scale + self.KLUB_Shift
        loss = self.compute_loss(images.clamp(0,1) * self.KLUB_Scale + self.KLUB_Shift, x0_hat, alpha, alpha_previous, eta, delta,logit_x_t)

        return loss                

    def compute_loss(self, x0, x0_hat, alpha, alpha_previous, eta, delta,logit_x_t):
        alpha_p = eta*delta*x0 
        beta_p = eta-eta*alpha_previous*x0
        alpha_q = eta*delta*x0_hat
        beta_q  = eta-eta*alpha_previous*x0_hat 

        _alpha_p = eta*alpha*x0 
        _beta_p  = eta-eta*alpha*x0
        _alpha_q = eta*alpha*x0_hat
        _beta_q  = eta-eta*alpha*x0_hat 

        KLUB_conditional = (self.KL_gamma(alpha_q,alpha_p).clamp(0)\
                                + self.KL_gamma(beta_q,beta_p).clamp(0)\
                                - self.KL_gamma(alpha_q+beta_q,alpha_p+beta_p).clamp(0)).clamp(0)
        KLUB_marginal = (self.KL_gamma(_alpha_q,_alpha_p).clamp(0)\
                            + self.KL_gamma(_beta_q,_beta_p).clamp(0)\
                            - self.KL_gamma(_alpha_q+_beta_q,_alpha_p+_beta_p).clamp(0)).clamp(0)

        
        KLUB_conditional_AS = (self.KL_gamma(alpha_p,alpha_q).clamp(0)\
                                    + self.KL_gamma(beta_p,beta_q).clamp(0)\
                                    - self.KL_gamma(alpha_p+beta_p,alpha_q+beta_q).clamp(0)).clamp(0)
        KLUB_marginal_AS = (self.KL_gamma(_alpha_p,_alpha_q).clamp(0)\
                                + self.KL_gamma(_beta_p,_beta_q).clamp(0)\
                                - self.KL_gamma(_alpha_p+_beta_p,_alpha_q+_beta_q).clamp(0)).clamp(0)
        #loss = KLUB_marginal
        loss_dict = {
            'KLUB': (.99 * KLUB_conditional + .01 * KLUB_marginal),
            'KLUB_marginal': KLUB_marginal,
            'KLUB_conditional': KLUB_conditional,
            'KLUB_AS': (0.99 * KLUB_conditional_AS + 0.01 * KLUB_marginal_AS),
            'ELBO': 0.99 * KLUB_conditional_AS + 0.01 * KLUB_marginal_AS,
            'KLUB_marginal_AS': KLUB_marginal_AS,
            'SNR_Weighted_ELBO': KLUB_marginal_AS,
            'KLUB_conditional_AS': KLUB_conditional_AS,
            'L2': torch.square(x0 - x0_hat),
            'L1': torch.abs(x0 - x0_hat),
            # Add other loss types here
        }

        if self.lossType not in loss_dict:
            raise NotImplementedError("Loss type not implemented")
        loss = loss_dict[self.lossType]
        
        return loss_dict[self.lossType]
        #return loss

    def log_gamma(self, alpha):
        #return torch.log(torch._standard_gamma(alpha).clamp(MIN))
        return torch.log(torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN))
        #return torch.log(torch._standard_gamma(alpha.to(torch.float64))).to(torch.float32)


    def KL_gamma(self, alpha_p, alpha_q, beta_p=None, beta_q=None):
        """
        Calculates the KL divergence between two Gamma distributions.
        alpha_p: the shape of the first Gamma distribution Gamma(alpha_p,beta_p).
        alpha_q: the shape of the second Gamma distribution Gamma(alpha_q,beta_q).
        beta_p (optional): the rate (inverse scale) of the first Gamma distribution Gamma(alpha_p,beta_p).
        beta_q (optional): the rate (inverse scale) of the second Gamma distribution Gamma(alpha_q,beta_q).
        """    
        KL = (alpha_p-alpha_q)*torch.digamma(alpha_p)-torch.lgamma(alpha_p)+torch.lgamma(alpha_q)
        if beta_p is not None and beta_q is not None:
            KL = KL + alpha_q*(torch.log(beta_p)-torch.log(beta_q))+alpha_p*(beta_q/beta_p-1.0)  
        return KL

    def KL_beta(self, alpha_p,beta_p,alpha_q,beta_q):
        """
        Calculates the KL divergence between two Beta distributions
        KL(Beta(alpha_p,beta_p) || Beta(alpha_q,beta_q))
        """
        KL =self.KL_gamma(alpha_p,alpha_q)+self.KL_gamma(beta_p,beta_q)-self.KL_gamma(alpha_p+beta_p,alpha_q+beta_q)
        return KL
    
def sigma_sigma(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    #return ((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1).sqrt()
    return (0.5 * beta_d * (t ** 2) + beta_min * t).expm1().sqrt()

def alpha_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return (-0.5 * beta_d * (t ** 2) - beta_min * t).exp()   

def log_alpha_log_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return -0.5 * beta_d * (t ** 2) - beta_min * t 

def bd_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=None, alpha_min=None, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, start_step=None,Scale=None,Shift=None,normalize_output = True
):
    
    if num_steps>350:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)

        t_steps = 1-step_indices / (num_steps - 1)*(1-1e-5)

        #t_steps = 0.996**(step_indices / (num_steps - 1)*500)

        # epsilon_s = 1e-5                
        # t_steps =  (epsilon_s + step_indices/(num_steps-1) * (1.0 - epsilon_s)).flip(dims=(0,))

        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        logit_alpha = sigmoid_start + (sigmoid_end-sigmoid_start) * (t_steps**sigmoid_power)
        alpha = logit_alpha.sigmoid()
    
    else:
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        log_pi = (sigmoid_start + (sigmoid_end - sigmoid_start) * (torch.tensor(1, dtype=torch.float64, device=latents.device)**sigmoid_power)).sigmoid().log() / (num_steps)
        #log_pi = torch.tensor(2e-6, device=latents.device).log()/(num_steps)
        alpha = (step_indices*log_pi).exp()
        alpha = torch.flip(alpha, [0])
        logit_alpha = alpha.logit()
        

    alpha_min = alpha_min if alpha_min is not None else alpha[0]

    if 1:
        log_u = log_gamma( (eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        log_v = log_gamma( (eta - eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        x_next = (log_u - log_v).to(latents.device)
    else:
        x_next = torch.logit( alpha_min*latents.to(torch.float64) ).to(latents.device)

    # if 1:
    #     x_next =x_next.clamp(torch.log( torch.tensor(MIN,device=latents.device)))

    # Main sampling loop.
    ims = []
    im_xhats = []
    for i, (logit_alpha_cur,logit_alpha_next) in enumerate(zip(logit_alpha[:-1], logit_alpha[1:])): # 0, ..., N-1
        x_cur = x_next
        alpha_cur = logit_alpha_cur.sigmoid()
        alpha_next = logit_alpha_next.sigmoid()

        log_alpha_cur = logsigmoid(logit_alpha_cur)

        xmin = Shift
        xmax = Shift + Scale
        xmean = Shift+Scale/2.0
        
        E1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).lgamma() - (eta * alpha_cur * xmin).lgamma())
        E2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).lgamma() - (eta-eta * alpha_cur * xmax).lgamma())
        E_logit_x_t =  E1 - E2


        V1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).digamma() - (eta * alpha_cur * xmin).digamma())
        V2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).digamma() - (eta-eta * alpha_cur * xmax).digamma())

        grids = (torch.arange(0,101,device=latents.device)/100 +0.5/100)*Scale+Shift
        grids = grids[:-1]
        alpha_x = alpha_cur*grids 
        if 1:
            #V3 = (((eta * alpha_cur * xmean).digamma())**2- E1**2).clamp(0)
            #V4 = (((eta-eta * alpha_cur * xmean).digamma())**2- E2**2).clamp(0)
            #V3 = E1**2
            #V4 = E2**2
            grids = (torch.arange(0,101,device=latents.device)/100)*Scale+Shift
            alpha_x = alpha_cur*grids 
            V3 = ((eta * alpha_x).digamma())**2
            #print(V3.shape)
            V3[0] = (V3[0]+V3[-1])/2
            V3 = V3[:-1]
            V3 = (V3.mean()- E1**2).clamp(0)   
            V4 = ((eta - eta*alpha_x).digamma())**2
            V4[0] = (V4[0]+V4[-1])/2
            V4 = V4[:-1]
            V4 = (V4.mean()- E2**2).clamp(0)
            
            #V3 = (( ((eta * alpha_x).digamma())**2).mean()- E1**2).clamp(0)           
            #V4 = (( ((eta - eta*alpha_x).digamma())**2).mean()- E2**2).clamp(0)
        else:
            V3 = (( ((eta * alpha_x).digamma())**2).mean()- E1**2).clamp(0)           
            V4 = (( ((eta - eta*alpha_x).digamma())**2).mean()- E2**2).clamp(0)


        std_logit_x_t = (V1+V2+V3+V4).sqrt()

        #std_logit_x_t = std_logit_x_t.clamp(EPS)

        logit_x_t = x_cur
        
        if class_labels is None:
            logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha_cur.unsqueeze(0).repeat(logit_x_t.shape[0])).to(torch.float64)
        else:    
            logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha_cur.unsqueeze(0).repeat(logit_x_t.shape[0]), class_labels).to(torch.float64)

        if normalize_output:

            variance = (1/4) * (torch.polygamma(1, eta * alpha_cur * (Scale + Shift)) - torch.polygamma(1, eta * (1- alpha_cur * (Scale+Shift)))) + \
                    (3/4) * (torch.polygamma(1, eta * alpha_cur * Shift) - torch.polygamma(1, eta * (1 - alpha_cur * Shift))) + \
                    (3/16) * ((torch.special.digamma(eta * alpha_cur * (Scale + Shift)) - torch.special.digamma(eta * (1 - alpha_cur * (Scale + Shift)))) - \
                                (torch.special.digamma(eta * alpha_cur * Shift) - torch.special.digamma(eta * (1 - alpha_cur * Shift))))**2

            std = torch.clamp(torch.sqrt(variance) * (eta * alpha_cur * Scale), max = 20)

            # mean = (1/4) * (torch.special.digamma(eta * alpha_cur * (Scale + Shift)) - torch.special.digamma(eta * (1- alpha_cur * (Scale+Shift)))) + \
            #         (3/4) * (torch.special.digamma(eta * alpha_cur * Shift) - torch.special.digamma(eta * (1 - alpha_cur * Shift)))

            # mean = mean * (eta * alpha_cur * Scale) - torch.lgamma(eta * alpha_cur * (Scale + Shift)) - torch.lgamma(eta * (1 - alpha_cur * (Scale + Shift))) + \
            #             torch.lgamma(eta * alpha_cur * (Shift)) + torch.lgamma(eta * (1 - alpha_cur * (Shift))) 

            # mean = torch.clamp(mean, min = -20)

            # logit_x0_hat = logit_x0_hat * std + mean
            logit_x0_hat = logit_x0_hat * std

        x0_hat = torch.sigmoid(logit_x0_hat)* Scale + Shift

        alpha_reverse = (eta*alpha_next-eta*alpha_cur)*x0_hat
        beta_reverse = eta-eta*alpha_next*x0_hat
        log_u = log_gamma(alpha_reverse.to(torch.float32)).to(torch.float64)
        log_v = log_gamma(beta_reverse.to(torch.float32)).to(torch.float64)
        concatenated = torch.cat((x_cur.unsqueeze(-1), (log_u-log_v).unsqueeze(-1), (x_cur+log_u-log_v).unsqueeze(-1)), dim=3)
        x_next = torch.logsumexp(concatenated, dim=3)    

    if 1: # step<num_steps/2:
        out = (x0_hat- Shift) / Scale
        out1 = ((torch.sigmoid(x_next)/alpha_next- Shift) / Scale) #.clamp(0,1)
        # else:
        #     out = 0.8*out+0.2*temp_out 
        #     #out1 = 0.8*out1+0.2*((torch.sigmoid(x_cur)/alpha[step]- Shift) / Scale) #.clamp(0,1)
        #     out1 = 0.8*out1+0.2*((torch.sigmoid(x_next)/alpha[step-1]- Shift) / Scale) #.clamp(0,1)
        #     #out1 = 0.8*out1+0.2*(torch.sigmoid(x_cur)/alpha[step]) #.clamp(0,1)    
    return out, out1  

def log_gamma(alpha):
    return torch.log(torch._standard_gamma(alpha))


if __name__ == '__main__':
    config = ModelParameters()
    args = get_args()
    print("args:", args)
    torch.set_default_dtype(torch.float32)

    """
    Sei model is published in the following paper
    Chen, K. M., Wong, A. K., Troyanskaya, O. G., & Zhou, J. (2022). A sequence-based global map of 
    regulatory activity for deciphering human genetics. Nature genetics, 54(7), 940-949. 
    [https://doi.org/10.1038/s41588-022-01102-2](https://doi.org/10.1038/s41588-022-01102-2)  
    """
    seifeatures = pd.read_csv(config.seifeatures_file, sep='|', header=None)

    sei = nn.DataParallel(NonStrandSpecific(Sei(4096, 21907)))
    sei.load_state_dict(torch.load(config.seimodel_file, map_location='cpu')['state_dict'])
    sei.cuda()

    train_set = TSSDatasetS(config, n_tsses=40000, rand_offset=10)
    data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers)

    #### PREPARE Valid DATASET
    valid_set = TSSDatasetS(config, split='valid', n_tsses=40000, rand_offset=0)
    valid_data_loader = DataLoader(valid_set, batch_size=config.val_batch_size, shuffle=False, num_workers=0)
    valid_datasets = []
    for x in valid_data_loader:
        valid_datasets.append(x)

    validseqs = []
    for seq in valid_datasets:
        validseqs.append(seq[:, :, :4])
    validseqs = np.concatenate(validseqs, axis=0)

    test_set = TSSDatasetS(config, split='test',n_tsses=40000, rand_offset=0)
    test_data_loader = DataLoader(test_set, batch_size=config.val_batch_size, shuffle=False, num_workers=0)
    test_datasets = []
    for x in test_data_loader:
        test_datasets.append(x)

    torch.set_default_dtype(torch.float32)

    with torch.no_grad():
        validseqs_pred = np.zeros((2915, 21907))
        for i in range(int(validseqs.shape[0] / 128)):
            validseq = validseqs[i * 128:(i + 1) * 128]
            validseqs_pred[i * 128:(i + 1) * 128] = sei(
                torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                           torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
        validseq = validseqs[-128:]
        validseqs_pred[-128:] = sei(
            torch.cat([torch.ones((validseq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(validseq).transpose(1, 2),
                       torch.ones((validseq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
    validseqs_predh3k4me3 = validseqs_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=1)

    SaveCheckPoint  = True

    #NFEs, the number of steps in reverse diffusion
    tqdm_epoch = tqdm.trange(config.num_epochs)

    score_model = nn.DataParallel(BetaDiffPrecond())
    score_model = score_model.to(config.device)
    score_model.train()

    optimizer = Adam(score_model.parameters(), lr=config.lr)

    bestsei_validloss = float('Inf')

    loss_fn = BetaDiffLoss(eta=10000.0, sigmoid_start=args.sigmoid_start, sigmoid_end=args.sigmoid_end, sigmoid_power=args.sigmoid_power, Scale=args.Scale, Shift=args.Shift, KLUB_Scale=args.KLUB_Scale, KLUB_Shift=args.KLUB_Shift, T=args.T, epsilon_t=1e-5,lossType=args.lossType, normalize_output = args.normalize_output)

    for epoch in tqdm_epoch:
        stime = time.time()

        for xS in (data_loader):
            x = xS[:, :, :4].to(config.device) #Data
            s = xS[:, :, 4:5].to(config.device) #Conditioning
            n_data = xS.shape[0]
            loss = loss_fn(net = score_model, images = x, labels=s)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:", epoch, "training loss:", loss.mean())

        if epoch % 5 == 0:
            score_model.eval()

            torch.set_default_dtype(torch.float32)
            allsamples = []
            # for t in tqdm.tqdm(valid_datasets):
            for t in valid_datasets:
                latents = torch.ones(size = (t.shape[0], t.shape[1], 4))*0.25
                allsamples.append(bd_sampler(net = score_model, latents = latents.cuda(), class_labels = t[:, :, 4:5].cuda(), randn_like=torch.randn_like, \
                num_steps=100, alpha_min=None, eta=args.eta, sigmoid_start=args.sigmoid_start, sigmoid_end=args.sigmoid_end, sigmoid_power=args.sigmoid_power, \
                    start_step=None,Scale=args.Scale,Shift=args.Shift, normalize_output = args.normalize_output)[0].detach().cpu().numpy())
                
            allsamples = np.concatenate(allsamples, axis=0)

            
            allsamples_pred = np.zeros((2915, 21907))
            for i in range(int(allsamples.shape[0] / 128)):
                seq = 1.0 * torch.nn.functional.one_hot(torch.argmax(torch.Tensor(allsamples[i * 128:(i + 1) * 128]), dim = 2), num_classes = 4)
                allsamples_pred[i * 128:(i + 1) * 128] = sei(
                    torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                            torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()
            seq = allsamples[-128:]
            allsamples_pred[-128:] = sei(
                torch.cat([torch.ones((seq.shape[0], 4, 1536)) * 0.25, torch.FloatTensor(seq).transpose(1, 2),
                        torch.ones((seq.shape[0], 4, 1536)) * 0.25], 2).cuda()).cpu().detach().numpy()

            allsamples_predh3k4me3 = allsamples_pred[:, seifeatures[1].str.strip().values == 'H3K4me3'].mean(axis=-1)
            valid_loss = ((validseqs_predh3k4me3 - allsamples_predh3k4me3) ** 2).mean()
            print(f"{epoch} valid sei loss {valid_loss} {time.time() - stime}", flush=True)
            if not args.debug:
                if valid_loss < bestsei_validloss:
                    print('Best valid SEI loss!')
                    bestsei_validloss = valid_loss
                    savepath = "".join(["sdedna_promoter_revision_beta_start_", str(args.sigmoid_start), \
                        "_end_", str(args.sigmoid_end), "_power_", str(args.sigmoid_power), "_eta_", str(args.eta), \
                            "_scale_", str(args.Scale), "_shift_", str(args.Shift), "_klub_scale_", str(args.KLUB_Scale), \
                                "_klub_shift_", str(args.KLUB_Shift), "_batch_size_", str(args.batch_size), ".sei.bestvalid.pth"])
                    torch.save(score_model.state_dict(), savepath)
            score_model.train()
            
    #Evaluate on test set using the best model chosen on validation set
    torch.set_default_dtype(torch.float32)

    score_model.load_state_dict(torch.load(savepath))

    allsamples = []
    for t in test_datasets:
        samples=[]
        for i in range(5):
            score_model.eval()
            latents = torch.ones(size = (t.shape[0], t.shape[1], 4))*0.25
            samples.append(bd_sampler(net = score_model, latents = latents.cuda(), class_labels = t[:, :, 4:5].cuda(), randn_like=torch.randn_like,
                    num_steps=100, alpha_min=None, eta=args.eta, sigmoid_start=args.sigmoid_start, sigmoid_end=args.sigmoid_end, sigmoid_power=args.sigmoid_power, start_step=None,Scale=args.Scale,Shift=args.Shift)[0].detach().cpu().numpy())
        allsamples.append(samples)
    
    allsamples = np.concatenate(allsamples, axis=1)
    testseqs = np.concatenate(test_datasets, axis=0)[:,:,:4]
    
    allsamples_pred = np.zeros((5, 2915, 21907))
    for j in range(5):
        for i in range(int(allsamples[j].shape[0]/128)):
            seq = 1.0* (allsamples[j][i*128:(i+1)*128] > 0.5)
            allsamples_pred[j,i*128:(i+1)*128] =  sei(torch.cat([torch.ones((seq.shape[0],4,1536))*0.25, torch.FloatTensor(seq).transpose(1,2), 
                    torch.ones((seq.shape[0],4,1536))*0.25], 2).cuda()).cpu().detach().numpy()
        seq = allsamples[j][-128:]
        allsamples_pred[j, -128:] =  sei(torch.cat([torch.ones((seq.shape[0],4,1536))*0.25, torch.FloatTensor(seq).transpose(1,2), 
                    torch.ones((seq.shape[0],4,1536))*0.25], 2).cuda()).cpu().detach().numpy()
        
    with torch.no_grad():
        testseqs_pred = np.zeros((2915, 21907))
        for i in range(int(testseqs.shape[0]/128)):
            testseq = testseqs[i*128:(i+1)*128]
            testseqs_pred[i*128:(i+1)*128] =  sei(torch.cat([torch.ones((testseq.shape[0],4,1536))*0.25, torch.FloatTensor(testseq).transpose(1,2), 
                    torch.ones((testseq.shape[0],4,1536))*0.25], 2).cuda()).cpu().detach().numpy()
        testseq = testseqs[-128:]
        testseqs_pred[-128:] =  sei(torch.cat([torch.ones((testseq.shape[0],4,1536))*0.25, torch.FloatTensor(testseq).transpose(1,2), 
                    torch.ones((testseq.shape[0],4,1536))*0.25], 2).cuda()).cpu().detach().numpy()

    testseqs_predh3k4me3 = testseqs_pred[:,seifeatures[1].str.strip().values=='H3K4me3'].mean(axis=1)
    exp=(10**np.concatenate(test_datasets, axis=0)[:,12:-12,4]-1).sum(axis=-1)
    allsamples_predh3k4me3 = allsamples_pred[:,:,seifeatures[1].str.strip().values=='H3K4me3'].mean(axis=-1)
    
    acc = []
    for i in range(5):
        acc.append(((allsamples_predh3k4me3[i] - testseqs_predh3k4me3)**2).mean())
    print("Results: ", np.mean(acc), np.std(acc)/np.sqrt(4))
