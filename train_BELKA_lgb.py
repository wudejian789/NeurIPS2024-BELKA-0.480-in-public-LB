import numpy as np
import pandas as pd
import lightgbm as lgb
import rdkit,argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import *
from sklearn.decomposition import PCA,IncrementalPCA

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--nnPCAdim', default=64, type=int)
parser.add_argument('--save_path', default='./saved_models/lightgbm.pkl')
parser.add_argument('--submit_file', default='./lgb_submission.csv')
parser.add_argument('--seed', default=9527, type=int)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--predict_only', default=False, type=bool)
parser.add_argument('--predict_output', type=str, default='')
parser.add_argument('--use_nn_list', default='unimollm;geminimol;unimol')
args = parser.parse_args()

def bitstring2uint8(bits):
    n = len(bits)//4
    return np.array([int(bits[i*4:(i+1)*4],2) for i in range(n)], np.uint8)
def uint82bitstring(uint8):
    return "".join([f"{bin(i)[2:]:0>4}" for i in uint8])

fpr2func = {'rdk': Chem.RDKFingerprint, # RDK指纹
            'Toplogical': Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect, # Toplogical指纹
            'MACCS': Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect, # MACCS指纹
            'AtomPair': Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect, # AtomPair指纹
            'ECFP': lambda x:AllChem.GetMorganFingerprintAsBitVect(x, 2), # ECFP指纹
            'FCFP': lambda x:AllChem.GetMorganFingerprintAsBitVect(x, 2, useFeatures=True), # FCFP指纹
            'Avalon': pyAvalonTools.GetAvalonFP, # Avalon指纹
            'Layered': Chem.rdmolops.LayeredFingerprint, # Layered指纹
            'Pattern': Chem.rdmolops.PatternFingerprint} # Pattern指纹

import importlib
class Wrapper:
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)
wrapped_mol_from_smiles = Wrapper("MolFromSmiles", "rdkit.Chem")

fpr2wrappedfunc = {'rdk': Wrapper("RDKFingerprint","rdkit.Chem"), # RDK指纹
                   'Toplogical': Wrapper("GetHashedTopologicalTorsionFingerprintAsBitVect","rdkit.Chem.rdMolDescriptors"), # Toplogical指纹
                   'MACCS': Wrapper("GetHashedTopologicalTorsionFingerprintAsBitVect","rdkit.Chem.rdMolDescriptors"), # MACCS指纹
                   'AtomPair': Wrapper("GetHashedAtomPairFingerprintAsBitVect","rdkit.Chem.rdMolDescriptors"), # AtomPair指纹
                   'ECFP4': lambda x:Wrapper("GetMorganFingerprintAsBitVect","rdkit.Chem.AllChem")(x, 2), # ECFP指纹
                   'FCFP6': lambda x:Wrapper("GetMorganFingerprintAsBitVect","rdkit.Chem.AllChem")(x, 3, useFeatures=True), # FCFP指纹
                   'Avalon': Wrapper("GetAvalonFP","rdkit.Avalon.pyAvalonTools"), # Avalon指纹
                   'Layered': Wrapper("LayeredFingerprint","rdkit.Chem.rdmolops"), # Layered指纹
                   'Pattern': Wrapper("PatternFingerprint","rdkit.Chem.rdmolops")} # Pattern指纹

def get_molecule_vector(smiList, func, convert2Mol=False, post_fn=None, usebar=True, parallel=False, njobs=2, dtype=np.float32):
    if usebar:
        smiList = tqdm(smiList)

    if parallel:
        from joblib import Parallel, delayed
        if convert2Mol:
            molList = Parallel(n_jobs=njobs)(delayed(wrapped_mol_from_smiles)(smi) for smi in smiList)
            if usebar: molList = tqdm(molList, position=0,leave=True)
            tmp = Parallel(n_jobs=njobs)(delayed(func)(mol) for mol in molList)
        else:
            tmp = Parallel(n_jobs=njobs)(delayed(func)(smi) for smi in smiList)
        if post_fn is not None:
            if usebar:
                tmp = tqdm(tmp, position=0,leave=True)
            tmp = Parallel(n_jobs=njobs)(delayed(getattr(i, post_fn))() for i in tmp)
    else:
        if convert2Mol:
            tmp = [func(Chem.MolFromSmiles(smi)) for smi in smiList]
        else:
            tmp = [func(smi) for smi in smiList]
        if post_fn is not None:
            tmp = [getattr(i, post_fn)() for i in tmp]
    return np.array(tmp, dtype=dtype)

if __name__ == '__main__':
    use_nn_list = set(args.use_nn_list.split(';'))
    SEED = int(args.seed)
    
    if args.restore is not None:
        with open(args.restore, 'rb') as f:
            restore = pickle.load(f)
    else:
        restore = None

    with open(f'dataset_all.pkl', 'rb') as f:
        totalDS = pickle.load(f)

    testDS = BELKA('./datasets/test.csv', protein_name=None, predict=True)

    fprArr,fprArr_test = [],[]
    for key in fpr2func:    
        fprArr.append( get_molecule_vector(totalDS.id2smiles, fpr2func[key], True, "ToList") )
        fprArr_test.append( get_molecule_vector(testDS.id2smiles, fpr2func[key], True, "ToList") )
    fprArr,fprArr_test = np.hstack(fprArr),np.hstack(fprArr_test)

    if restore is None:
        pca1 = PCA(n_components=64)
        pca1.fit(np.vstack([fprArr,fprArr_test]))
    else:
        pca1 = restore['fprPCA']

    fprFeaArr,fprFeaArr_test = pca1.transform(fprArr),pca1.transform(fprArr_test)

    if len(use_nn_list)>0:
        smi2fpr = []
        if 'unimollm' in use_nn_list:
            with open('./fpr_cache/unimollm_smi2fpr.pkl', 'rb') as f:
                smi2fpr.append( pickle.load(f) )
        if 'geminimol' in use_nn_list:
            with open('./fpr_cache/geminimol_smi2fpr.pkl', 'rb') as f:
                smi2fpr.append( pickle.load(f) )
        if 'unimol' in use_nn_list:
            with open('./fpr_cache/unimol_smi2fpr.pkl', 'rb') as f:
                smi2fpr.append( pickle.load(f) )

        nnFeaArr = np.array([np.hstack([smi2fpri[smi] for smi2fpri in smi2fpr]) for smi in totalDS.id2smiles], dtype=np.float32)
        nnFeaArr_test = np.array([np.hstack([smi2fpri[smi] for smi2fpri in smi2fpr]) for smi in testDS.id2smiles], dtype=np.float32)

        if restore is None:
            pca2 = PCA(n_components=int(args.nnPCAdim))
            pca2.fit(np.vstack([nnFeaArr,nnFeaArr_test]))
        else:
            pca2 = restore['nnPCA']

        nnFeaArr,nnFeaArr_test = pca2.transform(nnFeaArr),pca2.transform(nnFeaArr_test)

    from joblib import Parallel, delayed

    print('Get final training dataset for lgb...')
    if len(use_nn_list)>0 and os.path.exists('./fpr_cache/synsmiles_fea.mem'):
        totalX = np.zeros((len(totalDS),fprFeaArr.shape[1]*3+nnFeaArr.shape[1]*3), dtype=np.float32)
        for i in tqdm(range(len(totalDS))):
            totalX[i] = np.hstack([fprFeaArr[totalDS.sidList1[i]],fprFeaArr[totalDS.sidList2[i]],fprFeaArr[totalDS.sidList3[i]], #])
                                   nnFeaArr[totalDS.sidList1[i]],nnFeaArr[totalDS.sidList2[i]],nnFeaArr[totalDS.sidList3[i]]])
        totalX = np.hstack([totalX, np.memmap('./fpr_cache/synsmiles_fea.mem', dtype='float32', mode='r', shape=(len(totalDS),256))])

        testX = np.zeros((len(testDS),fprFeaArr.shape[1]*3+nnFeaArr.shape[1]*3), dtype=np.float32)
        for i in tqdm(range(len(testDS))):
            testX[i] = np.hstack([fprFeaArr_test[testDS.sidList1[i]],fprFeaArr_test[testDS.sidList2[i]],fprFeaArr_test[testDS.sidList3[i]], #])
                                  nnFeaArr_test[testDS.sidList1[i]],nnFeaArr_test[testDS.sidList2[i]],nnFeaArr_test[testDS.sidList3[i]]])
        testX = np.hstack([testX, np.memmap('./fpr_cache/synsmiles_test_fea.mem', dtype='float32', mode='r', shape=(len(testDS),256))])
    else:
        totalX = np.zeros((len(totalDS),fprFeaArr.shape[1]*3), dtype=np.float32)
        for i in tqdm(range(len(totalDS))):
            totalX[i] = np.hstack([fprFeaArr[totalDS.sidList1[i]],fprFeaArr[totalDS.sidList2[i]],fprFeaArr[totalDS.sidList3[i]]])
        
        testX = np.zeros((len(testDS),fprFeaArr.shape[1]*3), dtype=np.float32)
        for i in tqdm(range(len(testDS))):
            testX[i] = np.hstack([fprFeaArr_test[testDS.sidList1[i]],fprFeaArr_test[testDS.sidList2[i]],fprFeaArr_test[testDS.sidList3[i]]])
        
    totalY = np.array(totalDS.Y, dtype=np.float32)
    totalY1,totalY2,totalY3 = totalY[:,0],totalY[:,1],totalY[:,2]

    from sklearn import metrics as skmetrics
    params = {
        'boosting_type': 'gbdt',
        #'boosting': 'dart',
        'objective': 'binary',
        'metric': "None",
        # 'metric': 'map',
        'learning_rate': float(args.lr),
        'num_leaves': 127,
        'max_depth': -1,
        # 'max_bin': 10,
        # 'min_data_in_leaf': 8,
        # 'feature_fraction': 0.9,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        # 'lambda_l1': 0,
        # 'lambda_l2': 75,
        # 'min_split_gain': 0,
        # 'boost_from_average': False,
        # 'is_unbalance': True,
        # 'num_trees': 1,
        'verbose': 0
    }
    def feval_func(preds, train_data):
        # print(preds.shape, truth.Y.shape)
        return 'aupr', skmetrics.average_precision_score(train_data.get_label(),preds), True

    from lightgbm import log_evaluation, early_stopping
    callbacks = [log_evaluation(period=10), early_stopping(stopping_rounds=64)]

    from sklearn.model_selection import KFold,StratifiedKFold
    nfold = 15

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)
    predictY1,predictY2,predictY3 = np.zeros_like(totalY1),np.zeros_like(totalY2),np.zeros_like(totalY3)

    submitY = np.zeros((len(testX),3))

    if args.predict_only:
        for i,(trainIdx,validIdx) in enumerate(skf.split(range(totalX.shape[0]), y=totalY.sum(axis=1))):
            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY1[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY1[validIdx])
            
            gbm1 = restore['gbmList'][i][0]
            
            predictY1[validIdx] = gbm1.predict(totalX[validIdx])
            submitY[:,0] += gbm1.predict(testX) #/nfold
            
            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY2[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY2[validIdx])
            
            gbm2 = restore['gbmList'][i][1]

            predictY2[validIdx] = gbm2.predict(totalX[validIdx])
            submitY[:,1] += gbm2.predict(testX) #/nfold

            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY3[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY3[validIdx])
            
            gbm3 = restore['gbmList'][i][2]
            
            predictY3[validIdx] = gbm3.predict(totalX[validIdx])
            submitY[:,2] += gbm3.predict(testX) #/nfold
            break
        with open(args.predict_output.replace('valid','test'), 'wb') as f:
            pickle.dump({'y_prob':submitY}, f)
        with open(args.predict_output, 'wb') as f:
            pickle.dump({'y_prob':np.hstack([predictY1[validIdx].reshape(-1,1),predictY2[validIdx].reshape(-1,1),predictY3[validIdx].reshape(-1,1)]), 
                         'y_true':np.hstack([totalY1[validIdx].reshape(-1,1),totalY2[validIdx].reshape(-1,1),totalY3[validIdx].reshape(-1,1)])}, f)
    else:
        gbmList = []
        for i,(trainIdx,validIdx) in enumerate(skf.split(range(totalX.shape[0]), y=totalY.sum(axis=1))):
            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY1[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY1[validIdx])
            
            gbm1 = lgb.train(params, train_set=lgb_train, valid_sets=lgb_valid, num_boost_round=2560,
                            callbacks=callbacks, feval=feval_func)
            
            predictY1[validIdx] = gbm1.predict(totalX[validIdx])
            submitY[:,0] += gbm1.predict(testX)/nfold
            
            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY2[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY2[validIdx])
            
            gbm2 = lgb.train(params, train_set=lgb_train, valid_sets=lgb_valid, num_boost_round=2560,
                            callbacks=callbacks, feval=feval_func)

            predictY2[validIdx] = gbm2.predict(totalX[validIdx])
            submitY[:,1] += gbm2.predict(testX)/nfold

            lgb_train = lgb.Dataset(data=totalX[trainIdx], label=totalY3[trainIdx])
            lgb_valid = lgb.Dataset(data=totalX[validIdx], label=totalY3[validIdx])
            
            gbm3 = lgb.train(params, train_set=lgb_train, valid_sets=lgb_valid, num_boost_round=2560,
                            callbacks=callbacks, feval=feval_func)
            
            predictY3[validIdx] = gbm3.predict(totalX[validIdx])
            submitY[:,2] += gbm3.predict(testX)/nfold
            
            gbmList.append([gbm1,gbm2,gbm3])

            print(f'AUPR for each class in CV {i}:',
                  skmetrics.average_precision_score(totalY1[validIdx], predictY1[validIdx]),\
                  skmetrics.average_precision_score(totalY2[validIdx], predictY2[validIdx]),\
                  skmetrics.average_precision_score(totalY3[validIdx], predictY3[validIdx]))
            break

        with open(args.save_path, 'wb') as f:
            pickle.dump({'fprPCA':pca1, 'nnPCA':pca2, 'gbmList':gbmList}, f)

        print('generate the submission file...')
        df = pd.DataFrame([testDS.ids.reshape(-1),submitY.reshape(-1)]).T
        df.columns = ['id','binds']
        df['id'] = df['id'].astype('int32')
        df = df[df['id']>-1].reset_index(drop=True).sort_values(by=['id'])
        
        df.to_csv(args.submit_file, index=None)


