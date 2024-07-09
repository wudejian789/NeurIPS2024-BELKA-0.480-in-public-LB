import numpy as np
import pandas as pd
import torch,time,os,pickle,random,re,gc
from torch import nn as nn
from nnLayer import *
from metrics import *
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torch.distributed
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from joblib.externals.loky.backend.context import get_context
from utils import *
from tools import *

class BaseClassifier:
    def __init__(self, model):
        pass
    def calculate_y_logit(self, X):
        pass
    def calculate_y_prob(self, X):
        pass
    def calculate_y(self, X):
        pass
    def calculate_y_prob_by_iterator(self, dataStream):
        pass
    def calculate_y_by_iterator(self, dataStream):
        pass
    def calculate_loss(self, X, Y):
        pass
    def train(self, trainDataSet, validDataSet=None, otherDataSet=None, otherSampleNum=10000, doEvalTrain=False,
              batchSize=256, maxSteps=1000000, saveSteps=-1, evalSteps=100, earlyStop=10, beginSteps=-1,
              EMA=False, EMAdecay=0.9999, EMAse=16, scheduleLR=True,
              isHigherBetter=False, metrics="LOSS", report=["LOSS"], 
              savePath='model', shuffle=True, dataLoadNumWorkers=0, pinMemory=False, 
              ignoreIdx=-100, warmupSteps=0, SEED=0, tensorboard=False, prefetchFactor=2, bleuCommend=None):
        if self.DDP:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(f'./logs/{tensorboard}')
            self.writer = writer

        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        stop = False

        if scheduleLR:
            decaySteps = maxSteps - warmupSteps
            # schedulerRLR = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda i:i/warmupSteps if i<warmupSteps else (decaySteps-(i-warmupSteps))/decaySteps)
            schedulerRLR = get_cosine_schedule_with_warmup(self.optimizer, num_training_steps=maxSteps, num_warmup_steps=warmupSteps)

        if self.DDP:
            trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataSet, shuffle=True, seed=SEED)
        else:
            trainSampler = torch.utils.data.RandomSampler(trainDataSet)
        # trainSampler = torch.utils.data.RandomSampler(trainDataSet)
        trainStream = DataLoader(trainDataSet, batch_size=batchSize, collate_fn=self.collateFunc, sampler=trainSampler, drop_last=True,
                                 pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)

        if validDataSet is not None:
            if self.DDP:
                validSampler = torch.utils.data.distributed.DistributedSampler(validDataSet, shuffle=False)
            else:
                validSampler = torch.utils.data.SequentialSampler(validDataSet)
            # validSampler = torch.utils.data.SequentialSampler(validDataSet)
            validStream = DataLoader(validDataSet, batch_size=batchSize, collate_fn=self.collateFunc, sampler=validSampler, drop_last=False, 
                                     pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)       
     
        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        st = time.time()
        e,locStep = 0,-1
        ema = None

        # restore the state
        if beginSteps>0:
            for i in range(beginSteps):
                schedulerRLR.step()
            locStep = beginSteps

        while True:
            e += 1
            self.locEpoch = e
            if EMA and e>=EMAse and ema is None:
                print('Start EMA...')
                ema = EMAer(self.model, EMAdecay) # 0.9999
                ema.register()
            print(f"Preparing the epoch {e}'s data...")
            if hasattr(trainDataSet, 'init'): 
                trainDataSet.init(e)
            # if otherDataSet is not None:
            #    sampleIdx = random.sample(range(len(otherDataSet)), otherSampleNum)
            #    trainDS = torch.utils.data.ConcatDataset([trainDataSet, torch.utils.data.Subset(otherDataSet, sampleIdx)])
            #    if self.DDP:
            #        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDS, shuffle=True, seed=SEED)
            #    else:
            #        trainSampler = torch.utils.data.RandomSampler(trainDS)
            #    trainStream = DataLoader(trainDS, batch_size=batchSize, collate_fn=self.collateFunc, sampler=trainSampler, drop_last=True,
            #                             pin_memory=pinMemory, num_workers=dataLoadNumWorkers, prefetch_factor=prefetchFactor)
            if self.DDP:
                trainStream.sampler.set_epoch(e)
            pbar = tqdm(trainStream)
            self.to_train_mode()
            for data in pbar:
                data = dict_to_device(data, device=device)
                loss = self._train_step(data)
                # del data
                # gc.collect()
                if EMA and ema is not None:
                    ema.update()
                if scheduleLR:
                    schedulerRLR.step()
                if isinstance(loss, dict):
                    pbar.set_description(f"loss: { {k:round(float(loss[k]),3) for k in loss} }; lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}; Progress: {locStep/maxSteps:.3f}; Stop round: {stopSteps}")
                else:
                    pbar.set_description(f"loss: {loss:.3f}; lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}; Progress: {locStep/maxSteps:.3f}; Stop round: {stopSteps}")
                locStep += 1
                self.locStep = locStep
                if locStep>maxSteps:
                    print(f'Reach the max steps {maxSteps}... break...')
                    stop = True
                    break
                if (validDataSet is not None) and (locStep%evalSteps==0):
                    if EMA and ema is not None:
                        ema.apply_shadow()
                    print(f'========== Step:{locStep:5d} ==========')

                    self.to_eval_mode()
                    if doEvalTrain:
                        print(f'[Total Train]', end='')
                        if metrics == 'CUSTOMED':
                            mtc = self.calculate_CUSTOMED_metrics(trainStream)
                        else:
                            res = self.calculate_metrics_by_iterator(trainStream, metrictor, ignoreIdx, report)
                            metrictor.show_res(res)
                            mtc = res[metrics]

                    #data = self.calculate_y_prob_by_iterator(DataLoader(trainDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler))
                    #print(Y_pre.shape, Y.shape, type(Y_pre), type(Y))
                    #metrictor.set_data(data, ignoreIdx)
                    #print(f'[Total Train]',end='')
                    #metrictor(report)

                    print(f'[Total Valid]',end='')
                    if metrics == 'CUSTOMED':
                        mtc = self.calculate_CUSTOMED_metrics(validStream)
                    else:
                        res = self.calculate_metrics_by_iterator(validStream, metrictor, ignoreIdx, report)
                        metrictor.show_res(res)
                        mtc = res[metrics]
                    if self.DDP:
                        mtc = torch.tensor([mtc], dtype=torch.float32, device='cuda')
                        mtc_list = [torch.zeros_like(mtc) for i in range(world_size)]
                        torch.distributed.all_gather(mtc_list, mtc)
                        mtc = torch.cat(mtc_list).mean().detach().cpu().item()
                    if ((self.DDP and torch.distributed.get_rank() == 0) or not self.DDP):
                        # if tensorboard:
                        #     writer.add_scalar(metrics, mtc, locStep)
                        print('=================================')

                        if saveSteps>0 and locStep%saveSteps==0:
                            if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:
                                self.save("%s_acc%.3f_s%d.pkl"%(savePath,mtc,locStep), e+1, mtc)

                        if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                            if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:                
                                print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                                bestMtc = mtc
                                self.save("%s.pkl"%(savePath), e+1, bestMtc)
                            stopSteps = 0
                        else:
                            stopSteps += 1
                            if earlyStop>0 and stopSteps>=earlyStop:
                                print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                                stop = True
                                break

                        if EMA and ema is not None:
                            ema.restore()
                self.to_train_mode()
            if stop:
                break
        if tensorboard:
            writer.close()
        if (self.DDP and torch.distributed.get_rank() == 0) or not self.DDP:
            if EMA and ema is not None:
                ema.apply_shadow()
            self.load("%s.pkl"%savePath)
            self.to_eval_mode()
            os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, "%.3lf"%bestMtc))
            print(f'============ Result ============')
            if doEvalTrain:
                print(f'[Total Train]',end='')
                if metrics == 'CUSTOMED':
                    res = self.calculate_CUSTOMED_metrics(trainStream)
                else:
                    res = self.calculate_metrics_by_iterator(trainStream, metrictor, ignoreIdx, report)
                    metrictor.show_res(res)
            print(f'[Total Valid]',end='')
            if metrics == 'CUSTOMED':
                res = self.calculate_CUSTOMED_metrics(validStream)
            else:
                res = self.calculate_metrics_by_iterator(validStream, metrictor, ignoreIdx, report)
                metrictor.show_res(res)
            print(f'================================')
            return res
    def to_train_mode(self):
        self.model.train()  #set the module in training mode
        if self.collateFunc is not None:
            self.collateFunc.train = True
    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.train = False
    def _train_step(self, data):
        loss = self.calculate_loss(data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu().data.numpy()
    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        self.model.load_state_dict(parameters['model'])
        if 'optimizer' in parameters:
            self.optimizer.load_state_dict(parameters['optimizer'])
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

from torch.cuda.amp import autocast, GradScaler

class HuggingfaceSeq2SeqLanguageModel2_ForClassification(BaseClassifier):
    def __init__(self, model, criterion=None, optimizer=None, collateFunc=None, 
                 AMP=False, DDP=False, 
                 FGM=False, FGMeps=1., FGMse=-1,
                 metrictor=None,
                 multilabel=False):
        self.model = model
        self.criterion = (nn.CrossEntropyLoss() if not multilabel else nn.MultiLabelSoftMarginLoss()) if criterion is None else criterion
        self.optimizer = torch.optim.AdamW(model.parameters()) if optimizer is None else optimizer
        self.collateFunc = collateFunc
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        self.multilabel = multilabel
        if metrictor is not None:
            self.metrictor = metrictor
        if AMP:
            self.scaler = GradScaler()
        if FGM:
            self.fgm = FGMer(model, emb_name=['shared']) # word_embedding
            self.fgmEps = FGMeps
            self.FGMse = FGMse
    def _train_step(self, data):
        self.optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
                if isinstance(loss, dict):
                    loss_all = 0
                    for k in loss:
                        loss_all += loss[k]
                else:
                    loss_all = loss
            self.scaler.scale(loss_all).backward()
            if self.FGM and self.locEpoch >= self.FGMse: # 对抗损失
                self.fgm.attack(self.fgmEps)
                with autocast():
                    lossAdv = self.calculate_loss(data)
                self.scaler.scale(lossAdv).backward()
                self.fgm.restore()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            if isinstance(loss, dict):
                loss_all = 0
                for k in loss:
                    loss_all += loss[k]
            else:
                loss_all = loss
            loss_all.backward()
            if self.FGM:
                self.fgm.attack(self.fgmEps)
                lossAdv = self.calculate_loss(data)
                lossAdv.backward()
                self.fgm.fgm.restore()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
        return loss # .detach().cpu().data.numpy()
    def calculate_y_logit(self, data, predict=False):
        if self.AMP:
            with autocast():
                return self.model(data, predict)
        else:
            return self.model(data, predict)
    def calculate_loss(self, data):
        out = self.model(data)

        # return out['loss']
        if (len(data['y'].shape)==2 and data['y'].shape[1]==1 and not self.multilabel): Y = data['y'].reshape(-1).long()
        else: Y = data['y'].reshape(out['y_logit'].shape).long()
        if 'y_logit_list' not in out:
            Y_logit = out['y_logit'].reshape(len(Y),-1)
            return self.criterion(Y_logit, Y)
        else:
            loss = 0
            for y_logit in out['y_logit_list']:
                loss += self.criterion(y_logit.reshape(len(Y),-1), Y)
            # print(out['y_logit'].reshape(-1))
            # print(Y)
            # print()
            return loss
    def calculate_y_prob(self, data):
        tmp = self.model(data)
        if self.multilabel:
            return {'y_prob':F.sigmoid(tmp['y_logit'])}
        else:
            return {'y_prob':F.softmax(tmp['y_logit'], dim=-1)}
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_probArr = [],[]
        for data in tqdm(dataStream, position=0,leave=True):
            data = dict_to_device(data, device=device)
            Y_prob,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data,data['y'].detach().cpu().data
            YArr.append(Y)
            Y_probArr.append(Y_prob)

        YArr,Y_probArr = torch.cat(YArr, dim=0).numpy().astype('int32'),torch.cat(Y_probArr, dim=0).numpy().astype('float32')
        return {'y_prob':Y_probArr, 'y_true':YArr}
    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report):
        # if self.collateFunc is not None:
        #     self.collateFunc.train = True
        device = next(self.model.parameters()).device
        res = {}
        # if 'AUC' in report or 'MiAUC' in report or 'MaAUC' in report:
        res = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(res, ignore_index=ignoreIdx, multilabel=self.multilabel)
        return metrictor(report, isPrint=False)
        # else:
        #     cnt = 0
        #     for data in tqdm(dataStream):
        #         data = dict_to_device(data, device=device)
        #         with torch.no_grad():
        #             Y_pre,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy().astype('float32'),data['y'].detach().cpu().data.numpy().astype('int32')
        #             batchData = {'y_prob':Y_pre, 'y_true':Y}
        #             metrictor.set_data(batchData, ignore_index=ignoreIdx, multilabel=self.multilabel)
        #             batchRes = metrictor(report, isPrint=False)
        #         for k in batchRes:
        #             res.setdefault(k, 0)
        #             res[k] += batchRes[k]*len(Y)
        #         cnt += len(Y)
        # return {k:res[k]/cnt for k in res}
    def save(self, path, epochs, bestMtc=None):
        if self.DDP:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':model, 'optimizer':self.optimizer.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        if self.DDP:
            self.model.module.load_state_dict(parameters['model'])
        else:
            self.model.load_state_dict(parameters['model'])
        if 'optimizer' in parameters:
            try:
                self.optimizer.load_state_dict(parameters['optimizer'])
            except:
                print("Warning! Cannot restore the optimizer parameters...")
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))
        

# =================================================================================================
# DTI prediction
# =================================================================================================

class PseLabAttnDPI_forBELKA(nn.Module):
    def __init__(self, fHdnSizeList, hdnSize=256, fcSize=1024, pseLabNum=64, 
                 dropout=0.1, classNum=1, inputLayer='RNN',
                 fSize=2048):
        super(PseLabAttnDPI_forBELKA, self).__init__()
        self.atomFeaLN1 = nn.LayerNorm([75])
        self.atomFeaLN2 = nn.LayerNorm([75])
        self.atomFeaLN3 = nn.LayerNorm([75])
        self.atomFeaLN = nn.LayerNorm([75])

        self.dropout = nn.Dropout(p=dropout)
        
        self.atEmbedding = nn.Embedding(num_embeddings=850, embedding_dim=75)
        
        if inputLayer=='RNN':
            self.dLSTM1 = nn.LSTM(75, hdnSize//2, bidirectional=True, batch_first=True, num_layers=1, dropout=dropout)
            self.dLSTM2 = nn.LSTM(75, hdnSize//2, bidirectional=True, batch_first=True, num_layers=1, dropout=dropout)
            self.dLSTM3 = nn.LSTM(75, hdnSize//2, bidirectional=True, batch_first=True, num_layers=1, dropout=dropout)
            self.dLSTM = nn.LSTM(75, hdnSize//2, bidirectional=True, batch_first=True, num_layers=1, dropout=dropout)
        elif inputLayer=='CNN':
            self.dCNN1 = TextCNN(75, hdnSize//4, contextSizeList=[1,3,5,7])
            self.dCNN2 = TextCNN(75, hdnSize//4, contextSizeList=[1,3,5,7])
            self.dCNN3 = TextCNN(75, hdnSize//4, contextSizeList=[1,3,5,7])
            self.dCNN  = TextCNN(75, hdnSize//4, contextSizeList=[1,3,5,7])
        elif inputLayer=='Mamba':
            assert hdnSize==75
            from mambapy.mamba import Mamba,MambaConfig
            config = MambaConfig(d_model=hdnSize, n_layers=2)
            self.dMamba1 = Mamba(config)
            self.dMamba2 = Mamba(config)
            self.dMamba3 = Mamba(config)
            self.dMamba = Mamba(config)

        self.inputLayer = inputLayer

        self.dPseAttn1 = PseudoLabelAttention(hdnSize, pseLabNum, dropout=dropout)
        self.dPseAttn2 = PseudoLabelAttention(hdnSize, pseLabNum, dropout=dropout)
        self.dPseAttn3 = PseudoLabelAttention(hdnSize, pseLabNum, dropout=dropout)
        self.dPseAttn = PseudoLabelAttention(hdnSize, pseLabNum*2, dropout=dropout)
        
        self.fFcLinear1 = MLP(fSize, hdnSize, fHdnSizeList, outAct=True, name='fFcLinear', dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)
        self.fFcLinear2 = MLP(fSize, hdnSize, fHdnSizeList, outAct=True, name='fFcLinear', dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)
        self.fFcLinear3 = MLP(fSize, hdnSize, fHdnSizeList, outAct=True, name='fFcLinear', dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)
        self.fFcLinear = MLP(fSize, hdnSize, fHdnSizeList, outAct=True, name='fFcLinear', dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)

        self.crossLayers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hdnSize, nhead=4, batch_first=True), num_layers=2)

        self.ffn1 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*6),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*6, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*6)
                    )
        self.ffn2 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*6),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*6, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*6)
                    )
        self.classifier = nn.Sequential(
                            nn.BatchNorm1d(hdnSize*6),
                            nn.Dropout(dropout),
                            nn.Linear(hdnSize*6, classNum)
                        )
    def forward(self, data):
        batch1,batch2,batch3 = data['batch1'],data['batch2'],data['batch3']
        batch = data['batch']
        
        if self.inputLayer=='RNN':
            Xat1,_ = self.dLSTM1(self.dropout(torch.cat([self.atEmbedding(batch1['input_ids']),self.atomFeaLN1(data['atomFea1'])],dim=1))) # B,dL1,C
            Xat2,_ = self.dLSTM2(self.dropout(torch.cat([self.atEmbedding(batch2['input_ids']),self.atomFeaLN2(data['atomFea2'])],dim=1))) # B,dL2,C
            Xat3,_ = self.dLSTM3(self.dropout(torch.cat([self.atEmbedding(batch3['input_ids']),self.atomFeaLN3(data['atomFea3'])],dim=1))) # B,dL3,C
            Xat,_ = self.dLSTM(self.dropout(torch.cat([self.atEmbedding(batch['input_ids']),self.atomFeaLN(data['atomFea'])],dim=1))) # B,dL,C
        elif self.inputLayer=='CNN':
            Xat1 = torch.cat(self.dCNN1(self.dropout(torch.cat([self.atEmbedding(batch1['input_ids']),self.atomFeaLN1(data['atomFea1'])],dim=1))),dim=-1) # B,dL1,C
            Xat2 = torch.cat(self.dCNN2(self.dropout(torch.cat([self.atEmbedding(batch2['input_ids']),self.atomFeaLN2(data['atomFea2'])],dim=1))),dim=-1) # B,dL2,C
            Xat3 = torch.cat(self.dCNN3(self.dropout(torch.cat([self.atEmbedding(batch3['input_ids']),self.atomFeaLN3(data['atomFea3'])],dim=1))),dim=-1) # B,dL3,C
            Xat  = torch.cat(self.dCNN(self.dropout(torch.cat([self.atEmbedding(batch['input_ids']),self.atomFeaLN(data['atomFea'])],dim=1))),dim=-1) # B,dL,C
        elif self.inputLayer=='Mamba':
            Xat1 = self.dMamba1(self.dropout(torch.cat([self.atEmbedding(batch1['input_ids']),self.atomFeaLN1(data['atomFea1'])],dim=1))) # B,dL1,C
            Xat2 = self.dMamba2(self.dropout(torch.cat([self.atEmbedding(batch2['input_ids']),self.atomFeaLN2(data['atomFea2'])],dim=1))) # B,dL2,C
            Xat3 = self.dMamba3(self.dropout(torch.cat([self.atEmbedding(batch3['input_ids']),self.atomFeaLN3(data['atomFea3'])],dim=1))) # B,dL3,C
            Xat  = self.dMamba(self.dropout(torch.cat([self.atEmbedding(batch['input_ids']),self.atomFeaLN(data['atomFea'])],dim=1))) # B,dL,C

        Xat1 = self.dPseAttn1(Xat1, mask=torch.cat([batch1['attention_mask'],data['atom_mask1']], dim=1).bool()) # B,k,C
        Xat2 = self.dPseAttn2(Xat2, mask=torch.cat([batch2['attention_mask'],data['atom_mask2']], dim=1).bool()) # B,k,C
        Xat3 = self.dPseAttn3(Xat3, mask=torch.cat([batch3['attention_mask'],data['atom_mask3']], dim=1).bool()) # B,k,C
        Xat = self.dPseAttn(Xat, mask=torch.cat([batch['attention_mask'],data['atom_mask']], dim=1).bool()) # B,2*k,C
        
        X = torch.cat([Xat1,Xat2,Xat3,Xat], dim=1) # B,5*k,C
        
        X = self.crossLayers(X) # B,5*k,C

        X_max,_ = torch.max(X, dim=1) # B,C
        X_mean = torch.mean(X, dim=1) # B,C

        fgr1 = self.fFcLinear1(data['atomFin1']) # B,C
        fgr2 = self.fFcLinear2(data['atomFin2']) # B,C
        fgr3 = self.fFcLinear3(data['atomFin3']) # B,C
        fgr  = self.fFcLinear(data['atomFin']) # B,C

        out = torch.cat([X_max,X_mean,fgr1,fgr2,fgr3,fgr], dim=1) # B,C*6

        out = out+self.ffn1(out)
        out = out+self.ffn2(out)
        y_logit = self.classifier(out)
        return {'y_logit':y_logit}

class DeepFM_forBELKA(nn.Module):
    def __init__(self, fprNum=4, fprSize=8192, embSize=16, hdnList=[256,128], dropout=0.2):
        super(DeepFM_forBELKA, self).__init__()
        self.fm_1st_layer = nn.Linear(fprNum*fprSize, 3)
        
        self.fm_2nd_emb1 = nn.Linear(fprSize, embSize*3)
        self.fm_2nd_emb2 = nn.Linear(fprSize*4, embSize*3)

        self.dnn = MLP(fprSize*fprNum, 3, hdnList, dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)

        self.embSize = embSize
    def forward(self, data):
        X_sparse_emb = torch.cat([data['fprArr'].unsqueeze(1),\
                                  data['fprArr1'].unsqueeze(1),
                                  data['fprArr2'].unsqueeze(1),
                                  data['fprArr3'].unsqueeze(1)], dim=1) # B, 4, C
        X_dense_fea = torch.cat([data['fprArr'],data['fprArr1'],data['fprArr2'],data['fprArr3']], dim=1) # B, C
        B,C = X_dense_fea.shape

        # FM一阶部分
        fm_1st_part = self.fm_1st_layer(X_dense_fea) # B,3

        # FM二阶部分
        fm_2nd_vec = self.fm_2nd_emb1(X_sparse_emb).reshape(B,4,self.embSize,3) # B,4,embSize,3
        fm_2st_part1 = 0.5 * torch.sum(torch.sum(fm_2nd_vec,dim=1)**2 - torch.sum(fm_2nd_vec**2, dim=1), dim=1) # B,3

        fm_2nd_vec = self.fm_2nd_emb2(X_dense_fea).reshape(B,self.embSize,3) # B,embSize,3
        fm_2st_part2 = 0.5 * (torch.sum(fm_2nd_vec,dim=1)**2 - torch.sum(fm_2nd_vec**2, dim=1)) # B,3

        # DNN部分
        dnn_part = self.dnn(X_dense_fea) # B,C
        return {'y_logit':fm_1st_part+fm_2st_part1+fm_2st_part2+dnn_part}

class DeepFM2_forBELKA(nn.Module):
    def __init__(self, fprNum=4, fprSize=8192, embSize=16, hdnList=[256,128], dropout=0.2):
        super(DeepFM2_forBELKA, self).__init__()
        self.fm_1st_layer = nn.Linear(fprNum*fprSize, embSize)

        self.fm_2nd_emb1 = nn.Linear(fprSize, embSize)
        self.fm_2nd_emb2 = nn.Linear(fprSize*4, embSize*embSize)

        self.dnn = MLP(3*embSize, 3, hdnList, dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)

        self.embSize = embSize
    def forward(self, data):
        X_sparse_emb = torch.cat([data['fprArr'].unsqueeze(1),\
                                  data['fprArr1'].unsqueeze(1),
                                  data['fprArr2'].unsqueeze(1),
                                  data['fprArr3'].unsqueeze(1)], dim=1) # B, 4, C
        X_dense_fea = torch.cat([data['fprArr'],data['fprArr1'],data['fprArr2'],data['fprArr3']], dim=1) # B, C
        B,C = X_dense_fea.shape

        # FM一阶部分
        fm_1st_part = self.fm_1st_layer(X_dense_fea) # B,embSize

        # FM二阶部分
        fm_2nd_vec = self.fm_2nd_emb1(X_sparse_emb) # B,4,embSize
        fm_2st_part1 = 0.5 * (torch.sum(fm_2nd_vec,dim=1)**2 - torch.sum(fm_2nd_vec**2, dim=1)) # B,embSize

        fm_2nd_vec = self.fm_2nd_emb2(X_dense_fea).reshape(B,self.embSize,self.embSize) # B,embSize,embSize
        fm_2st_part2 = 0.5 * (torch.sum(fm_2nd_vec,dim=1)**2 - torch.sum(fm_2nd_vec**2, dim=1)) # B,embSize

        # DNN部分
        return {'y_logit':self.dnn(torch.cat([fm_1st_part,fm_2st_part1,fm_2st_part2],dim=1))} # B,C

class FprMLP_forBELKA(nn.Module):
    def __init__(self, fprSize=8192, hdnSize=2048, dropout=0.2):
        super(FprMLP_forBELKA, self).__init__()

        self.ffn1 = nn.Sequential(
                        nn.BatchNorm1d(fprSize),
                        nn.Dropout(dropout),
                        nn.Linear(fprSize, hdnSize),
                        nn.BatchNorm1d(hdnSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize,fprSize)
                    )
        self.ffn2 = nn.Sequential(
                        nn.BatchNorm1d(fprSize),
                        nn.Dropout(dropout),
                        nn.Linear(fprSize, hdnSize),
                        nn.BatchNorm1d(hdnSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize,fprSize)
                    )
        self.classifier = nn.Sequential(
                            nn.BatchNorm1d(fprSize),
                            nn.Dropout(dropout),
                            nn.Linear(fprSize, 3)
                        )

    def forward(self, data):
        out = torch.cat([data['fprArr'],data['fprArr1'],data['fprArr2'],data['fprArr3']], dim=1) # B, C
        
        out = out+self.ffn1(out)
        out = out+self.ffn2(out)
        y_logit = self.classifier(out)
        return {'y_logit':y_logit}

class FprMLP2_forBELKA(nn.Module):
    def __init__(self, fprSize=8192, hdnSize=128, fcSize=1024, dropout=0.2):
        super(FprMLP2_forBELKA, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.tknEmbedding = nn.Embedding(num_embeddings=850, embedding_dim=hdnSize)
        
        # from mambapy.mamba import Mamba,MambaConfig
        # config = MambaConfig(d_model=hdnSize, n_layers=2)
        # self.tknMamba = Mamba(config)
        self.tknCNN  = TextCNN(hdnSize, hdnSize//4, contextSizeList=[1,3,5,7], reduction='pool')

        self.fFcLinear = MLP(fprSize*3, hdnSize, [hdnSize], outAct=True, dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)
        
        hdnSize *= 2
        self.ffn1 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize)
                    )
        self.ffn2 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize)
                    )
        self.classifier = nn.Sequential(
                            nn.BatchNorm1d(hdnSize),
                            nn.Dropout(dropout),
                            nn.Linear(hdnSize, 3)
                        )

    def forward(self, data):
        batch = data['batch']
        
        # Xsmi = torch.max(self.tknMamba(self.dropout(self.tknEmbedding(batch['input_ids']))),dim=1)[0] # B,dL,C
        Xsmi = self.tknCNN(self.dropout(self.tknEmbedding(batch['input_ids']))) # B,C

        Xfgr = self.fFcLinear(torch.cat([data['fprArr1'],data['fprArr2'],data['fprArr3']], dim=1)) # B, C
        
        out = torch.cat([Xsmi,Xfgr], dim=1) # B,C
        out = out+self.ffn1(out)
        out = out+self.ffn2(out)
        y_logit = self.classifier(out)
        return {'y_logit':y_logit}

class GraphMLP_forBELKA(nn.Module):
    def __init__(self, hdnSize=128, fcSize=1024, hdnList=[128,128], dropout=0.1):
        super(GraphMLP_forBELKA, self).__init__()

        self.atomFeaLN1 = nn.LayerNorm([75])
        self.atomFeaLN2 = nn.LayerNorm([75])
        self.atomFeaLN3 = nn.LayerNorm([75])
        self.atomFeaLN = nn.LayerNorm([75])

        self.dropout = nn.Dropout(p=dropout)
        self.atEmbedding = nn.Embedding(num_embeddings=850, embedding_dim=hdnSize-75)

        self.molGCN = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN', \
                          dropout=dropout, dpEveryLayer=True, \
                          outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN1 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN1', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN2 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN2', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN3 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN3', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)

        self.ffn1 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*4),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*4, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*4)
                    )
        self.ffn2 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*4),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*4, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*4)
                    )
        self.classifier = nn.Sequential(
                            nn.BatchNorm1d(hdnSize*4),
                            nn.Dropout(dropout),
                            nn.Linear(hdnSize*4, 3)
                        )
    def forward(self, data):
        batch1,batch2,batch3 = data['batch1'],data['batch2'],data['batch3']
        batch = data['batch']

        X = self.dropout(torch.cat([self.atEmbedding(batch['input_ids']),self.atomFeaLN1(data['atomFea'])],dim=2))    # B,L,C
        X1 = self.dropout(torch.cat([self.atEmbedding(batch1['input_ids']),self.atomFeaLN1(data['atomFea1'])],dim=2)) # B,L,C
        X2 = self.dropout(torch.cat([self.atEmbedding(batch2['input_ids']),self.atomFeaLN1(data['atomFea2'])],dim=2)) # B,L,C
        X3 = self.dropout(torch.cat([self.atEmbedding(batch3['input_ids']),self.atomFeaLN1(data['atomFea3'])],dim=2)) # B,L,C

        X = self.molGCN(X, data['atomAdj'])     # B,L,C
        X1 = self.molGCN1(X1, data['atomAdj1']) # B,L,C
        X2 = self.molGCN2(X2, data['atomAdj2']) # B,L,C
        X3 = self.molGCN3(X3, data['atomAdj3']) # B,L,C

        out = torch.cat([X.max(dim=1)[0],X1.max(dim=1)[0],X2.max(dim=1)[0],X3.max(dim=1)[0]], dim=1) # B,C
        out = out+self.ffn1(out)
        out = out+self.ffn2(out)
        y_logit = self.classifier(out)
        return {'y_logit':y_logit}

class GraphMLP2_forBELKA(nn.Module):
    def __init__(self, fHdnSizeList=[256], hdnSize=128, fcSize=2048, hdnList=[128,128,128,128], dropout=0.1):
        super(GraphMLP2_forBELKA, self).__init__()

        self.atomFeaLN1 = nn.LayerNorm([75])
        self.atomFeaLN2 = nn.LayerNorm([75])
        self.atomFeaLN3 = nn.LayerNorm([75])
        self.atomFeaLN = nn.LayerNorm([75])

        self.dropout = nn.Dropout(p=dropout)
        self.atEmbedding = nn.Embedding(num_embeddings=850, embedding_dim=hdnSize-75)

        self.molGCN = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN', \
                          dropout=dropout, dpEveryLayer=True, \
                          outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN1 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN1', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN2 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN2', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)
        self.molGCN3 = GCN(hdnSize, hdnSize, hdnList, name='nodeGCN3', \
                           dropout=dropout, dpEveryLayer=True, \
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=True)

        self.fFcLinear = MLP(2048*4, hdnSize*2, fHdnSizeList, outAct=True, name='fFcLinear', dropout=dropout, dpEveryLayer=True, bnEveryLayer=True, inBn=True, inDp=True)

        self.ffn1 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*6),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*6, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*6)
                    )
        self.ffn2 = nn.Sequential(
                        nn.BatchNorm1d(hdnSize*6),
                        nn.Dropout(dropout),
                        nn.Linear(hdnSize*6, fcSize),
                        nn.BatchNorm1d(fcSize),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(fcSize,hdnSize*6)
                    )
        self.classifier = nn.Sequential(
                            nn.BatchNorm1d(hdnSize*6),
                            nn.Dropout(dropout),
                            nn.Linear(hdnSize*6, 3)
                        )
    def forward(self, data):
        batch1,batch2,batch3 = data['batch1'],data['batch2'],data['batch3']
        batch = data['batch']

        X = self.dropout(torch.cat([self.atEmbedding(batch['input_ids']),self.atomFeaLN1(data['atomFea'])],dim=2))    # B,L,C
        X1 = self.dropout(torch.cat([self.atEmbedding(batch1['input_ids']),self.atomFeaLN1(data['atomFea1'])],dim=2)) # B,L,C
        X2 = self.dropout(torch.cat([self.atEmbedding(batch2['input_ids']),self.atomFeaLN1(data['atomFea2'])],dim=2)) # B,L,C
        X3 = self.dropout(torch.cat([self.atEmbedding(batch3['input_ids']),self.atomFeaLN1(data['atomFea3'])],dim=2)) # B,L,C

        X = self.molGCN(X, data['atomAdj'])     # B,L,C
        X1 = self.molGCN1(X1, data['atomAdj1']) # B,L,C
        X2 = self.molGCN2(X2, data['atomAdj2']) # B,L,C
        X3 = self.molGCN3(X3, data['atomAdj3']) # B,L,C

        fgr  = self.fFcLinear(torch.cat([data['atomFin'],data['atomFin1'],data['atomFin2'],data['atomFin3']], dim=1)) # B,C

        out = torch.cat([X.max(dim=1)[0],X1.max(dim=1)[0],X2.max(dim=1)[0],X3.max(dim=1)[0],fgr], dim=1) # B,C
        out = out+self.ffn1(out)
        out = out+self.ffn2(out)
        y_logit = self.classifier(out)
        return {'y_logit':y_logit}


class BridgeDPI(nn.Module):
    def __init__(self, outSize, 
                 cHiddenSizeList, 
                 fHiddenSizeList, 
                 fSize=1024, cSize=8420,
                 gcnHiddenSizeList=[], fcHiddenSizeList=[], nodeNum=32, resnet=True,
                 hdnDropout=0.1, fcDropout=0.2,  
                 useFeatures = {"kmers":True,"pSeq":True,"FP":True,"dSeq":True}, 
                 maskDTI=False):
        super(BridgeDPI, self).__init__()
        self.nodeEmbedding = TextEmbedding(torch.tensor(np.random.normal(size=(max(nodeNum,0),outSize)), dtype=torch.float32), tknDropout=hdnDropout/2, embDropout=hdnDropout/2, name='nodeEmbedding')
            
        self.amEmbedding = TextEmbedding(torch.eye(40), tknDropout=hdnDropout/2, embDropout=hdnDropout/2, freeze=True, name='amEmbedding')
        self.pCNN = TextCNN(40, 64, [25], ln=True, name='pCNN', reduction='pool')
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='pFcLinear')

        self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN', reduction='pool')
        self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True, outAct=True, outDp=True, name='dFcLinear')

        self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear', dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, inBn=True, outBn=True)
        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear', dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, inBn=True, outBn=True)
        
        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet)
        
        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True, dpEveryLayer=True)

        self.nodeNum = nodeNum
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

    def forward(self, X):
        Xam = (self.cFcLinear(X['aminoCtr']).unsqueeze(1) if self.useFeatures['kmers'] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(X['cbatch']['input_ids']))).unsqueeze(1) if self.useFeatures['pSeq'] else 0) # => batchSize × 1 × outSize
        Xat = (self.fFcLinear(X['atomFin']).unsqueeze(1) if self.useFeatures['FP'] else 0) + \
              (self.dFcLinear(self.dCNN(X['atomFea'])).unsqueeze(1) if self.useFeatures['dSeq'] else 0) # => batchSize × 1 × outSize

        if self.nodeNum>=0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(len(Xat), 1, 1)
            node = torch.cat([Xam, Xat, node], dim=1) # => batchSize × nodeNum × outSize
            nodeDist = torch.sqrt(torch.sum(node**2,dim=2,keepdim=True)+1e-8)# => batchSize × nodeNum × 1 

            cosNode = torch.matmul(node,node.transpose(1,2)) / (nodeDist*nodeDist.transpose(1,2)+1e-8) # => batchSize × nodeNum × nodeNum
            # cosNode = cosNode*0.5 + 0.5
            # cosNode = F.relu(cosNode) # => batchSize × nodeNum × nodeNum
            cosNode[cosNode<0] = 0
            cosNode[:,range(node.shape[1]),range(node.shape[1])] = 1 # => batchSize × nodeNum × nodeNum
            if self.maskDTI: 
                cosNode[:,0,1] = cosNode[:,1,0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=node.device).repeat(len(Xam),1,1) # => batchSize × nodeNum × nodeNum
            D[:,range(node.shape[1]),range(node.shape[1])] = 1/(torch.sum(cosNode,dim=2)**0.5)
            pL = torch.matmul(torch.matmul(D,cosNode),D) # => batchSize × batchnodeNum × nodeNumSize
            node_gcned = self.nodeGCN(node, pL) # => batchSize × nodeNum × outSize

            node_embed = node_gcned[:,0,:]*node_gcned[:,1,:] # => batchSize × outSize
        else:
            node_embed = (Xam*Xat).squeeze(dim=1) # => batchSize × outSize
        return {"y_logit":self.fcLinear(node_embed)}#, "loss":1*l2}

