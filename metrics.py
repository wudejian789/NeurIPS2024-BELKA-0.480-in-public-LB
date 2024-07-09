import numpy as np
from sklearn import metrics as skmetrics
import torch
import warnings
warnings.filterwarnings("ignore")

def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True
    
class Metrictor:
    def __init__(self, gama=0.2):
        self._reporter_ = {"ACC":self.ACC, "AUC":self.AUC, "LOSS":self.LOSS, 
                           "MSE":self.MSE, "RMSE":self.RMSE, "MAE":self.MAE, 
                           "Pearson":self.Pearson, "CI":self.CI, "R2m":self.R2m,
                           "R2":self.R2, "MaAUC":self.MaAUC, "MiAUPR":self.MiAUPR, "MiAUC":self.MiAUC, "MaAUPR":self.MaAUPR,
                           "ValidMaAUC":self.ValidMaAUC, "ValidMaAUPR":self.ValidMaAUPR, "ValidACC":self.ValidACC, 
                           "RMSD":self.RMSD}
        self.gama = gama
    def __call__(self, report, end='\n', isPrint=True):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            if isPrint: print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        if isPrint: print(end=end)
        return res
    
    def show_res(self, res, end='\n'):
        for mtc in res:
            v = res[mtc]
            print(f" {mtc}={v:6.3f}", end=';')
        print(end=end)

    def set_data(self, data, ignore_index=None, multilabel=False, xyz=False):
        if multilabel:
            self.Y = data['y_true']
            if 'y_prob' in data:
                self.Y_prob_pre = data['y_prob'].reshape(self.Y.shape)
                self.Y_pre = self.Y_prob_pre>0.5
            else:
                self.Y_pre = data['y_pre'].reshape(self.Y.shape)
        elif xyz:
            self.Y = data['y_true'].reshape(-1,3)
            self.Y_pre = data['y_pre'].reshape(-1,3)
        else:
            yTrue = data['y_true'].reshape(-1)

            isValid = np.ones(len(yTrue), dtype=bool)
            if ignore_index is not None:
                isValid &= yTrue!=ignore_index
            if 'mask_idx' in data:
                isValid &= data['mask_idx'].reshape(-1)
            self.Y = yTrue[ isValid ]

            if 'y_prob' in data:
                yProb = data['y_prob'].reshape(len(yTrue), -1)
                self.Y_prob_pre = yProb[ isValid ]
                self.Y_pre = self.Y_prob_pre.argmax(axis=-1)
            else:
                yPre = data['y_pre'].reshape(-1)
                self.Y_pre = yPre[ isValid ]

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')

    def ACC(self):
        return np.mean(self.Y_pre==self.Y)
    def AUC(self):
        if len(self.Y_prob_pre.shape)==2 and len(self.Y.shape)==1:
            return skmetrics.roc_auc_score(self.Y, self.Y_prob_pre[:,1])
        else:
            return skmetrics.roc_auc_score(self.Y, self.Y_prob_pre)
    def MaAUC(self):
        try:
            return skmetrics.roc_auc_score(self.Y, self.Y_prob_pre, average='macro')
        except:
            tmp = [skmetrics.roc_auc_score(self.Y[:,i],self.Y_prob_pre[:,i]) for i in range(self.Y.shape[1]) if len(set(self.Y[:,i]))>1]
            return np.mean(tmp)
    def MaAUPR(self):
        tmp = [skmetrics.average_precision_score(self.Y[:,i],self.Y_prob_pre[:,i]) for i in range(self.Y.shape[1]) if len(set(self.Y[:,i]))>1]
        print('each AUPR:', tmp)
        return np.mean(tmp)
    def ValidMaAUC(self):
        aucList = []
        for i in range(self.Y.shape[1]):
            if np.sum(self.Y[:,i]==1)>0 and np.sum(self.Y[:,i]==0)>0:
                # ignore nan values
                isValid = self.Y[:,i]>-0.5
                aucList.append( skmetrics.roc_auc_score(self.Y[isValid, i], self.Y_prob_pre[isValid, i]) )
        if len(aucList) < self.Y.shape[1]:
            print('Some targets are missing...')
        return np.mean(aucList)
    def ValidMaAUPR(self):
        auprList = []
        for i in range(self.Y.shape[1]):
            if np.sum(self.Y[:,i]==1)>0 and np.sum(self.Y[:,i]==0)>0:
                # ignore nan values
                isValid = self.Y[:,i]>-0.5
                auprList.append( skmetrics.average_precision_score(self.Y[isValid, i], self.Y_prob_pre[isValid, i]) )
        if len(auprList) < self.Y.shape[1]:
            print('Some targets are missing...')
        return np.mean(auprList)
    def ValidACC(self):
        isValid = self.Y>0.5
        return (self.Y_pre==self.Y)[isValid].sum() / isValid.sum()
    def MiAUC(self):
        return skmetrics.roc_auc_score(self.Y, self.Y_prob_pre, average='micro')
    def MiAUPR(self):
        return skmetrics.average_precision_score(self.Y, self.Y_prob_pre, average='micro')
    def LOSS(self):
        return LOSS(self.Y_prob_pre,self.Y)
    def MSE(self):
        return np.mean((self.Y_pre-self.Y)**2)
    def RMSE(self):
        if len(self.Y_pre.shape)==2:
            print([np.sqrt(np.mean((self.Y_pre[:,i]-self.Y[:,i])**2)) for i in range(self.Y_pre.shape[1])])
        return np.sqrt(self.MSE())
    def RMSD(self):
        return np.mean(np.sqrt(np.sum((self.Y_pre-self.Y)**2, axis=-1)))
    def MAE(self):
        return np.mean(np.abs(self.Y_pre-self.Y))
    def Pearson(self):
        return (np.mean(self.Y_pre*self.Y) - np.mean(self.Y_pre)*np.mean(self.Y)) / np.sqrt((np.mean(self.Y_pre**2)-np.mean(self.Y_pre)**2) * (np.mean(self.Y**2)-np.mean(self.Y)**2))
    def R2(self):
        return self.Pearson()**2
    def CI(self):
        return CI(self.Y_pre, self.Y)
    def R2m(self):
        return get_rm2(self.Y_pre, self.Y)

def CI(yp,yt):
    ind = np.argsort(yt)
    yp,yt = yp[ind],yt[ind]
    yp_diff = yp.reshape(1,-1) - yp.reshape(-1,1)
    yt_diff = yt.reshape(1,-1) - yt.reshape(-1,1)
    yp_diff,yt_diff = np.triu(yp_diff,1),np.triu(yt_diff,1)
    tmp = yp_diff[yt_diff>0].reshape(-1)
    return (1.0*(tmp>0).sum() + 0.5*(tmp==0).sum()) / len(tmp)

#R2m
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)
def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
def get_rm2(ys_line, ys_orig):
    r2 = r_squared_error(ys_orig, ys_line) # 约等于pearson的平方
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def LOSS(Y_prob_pre, Y):
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    #return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))

    if isinstance(Y_prob_pre, torch.Tensor):
        return -torch.log(Y_prob_pre[range(len(Y)),Y]).mean()
    else:
        return -np.log(Y_prob_pre[range(len(Y)),Y]).mean()
    
def ContrastiveLOSS(vectors, labels, gama=0.2):
    dist = vectors.dot(vectors.T) # => B × B
    dist = np.exp(dist/gama)
    loss = 0

    for lab in set(labels):
        posIdx = labels==lab
        tmp = dist[posIdx]
        pos = tmp[:,posIdx][~np.eye(len(tmp), dtype='bool')].reshape(len(tmp),-1)
        neg = np.sum(tmp[:,~posIdx], axis=1, keepdims=True)
        loss += -np.sum(np.mean(np.log2(pos / (pos+neg)), axis=1))
    return loss / len(labels)
