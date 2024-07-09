# from utils import *
# import pickle
# totalDS = BELKA('./datasets/train.csv', protein_name=None)
# with open(f'dataset_all_new.pkl', 'wb') as f:
#     pickle.dump(totalDS, f)

import os,random
os.environ["TOKENIZERS_PARALLELISM"] = "true"
SEED = 9527

import pandas as pd
import numpy as np

from utils import *
from DL_ClassifierModel import *
import pickle,argparse,datetime

parser = argparse.ArgumentParser()
parser.add_argument('--modelType', default='FprMLP', type=str)
# parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--batchSize', default=1280, type=int)
# parser.add_argument('--seqMaxLen', default=160, type=int)
parser.add_argument('--epochs', default=6, type=int)
parser.add_argument('--earlyStop', default=9999, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weightDecay', default=0.1, type=float)
parser.add_argument('--warmupRatio', default=0.06, type=float)
parser.add_argument('--ddp', default='false', type=str)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--beginSteps', default=None, type=float)
parser.add_argument('--split_type', default='str', type=str)
parser.add_argument('--seed', default=9527, type=int)
parser.add_argument('--predict_only', default=False, type=bool)
parser.add_argument('--predict_output', type=str, default='')
parser.add_argument('--EMA', default=False, type=bool)
parser.add_argument('--EMAse', default=6, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    if args.split_type=='full':
        SEED = int(random.random()*1000000)
        set_seed(SEED)
        print('SEED:', SEED)
    else:
        SEED = int(args.seed)
        set_seed(SEED)

    ddp = args.ddp=='true'
    print(os.system('hostname'))
    if ddp:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank) # 设定cuda的默认GPU，每个rank不同
        print('local_rank:', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000)) # , world_size=args.dataLoadNumWorkers
    

    # totalDS = BELKA('./datasets/train.csv', protein_name=None)
    # with open(f'dataset_all_new.pkl', 'wb') as f:
    #     pickle.dump(totalDS, f)
    with open(f'dataset_all.pkl', 'rb') as f:
        totalDS = pickle.load(f)
    totalDS.id2smiles = np.array(totalDS.id2smiles)
    totalDS.id2mol = np.array(totalDS.id2mol)
    totalDS.id2cansmiles = np.array(totalDS.id2cansmiles)
    totalDS.id2dFinData = np.array(totalDS.id2dFinData)
    totalDS.sidList1 = np.array(totalDS.sidList1, dtype=np.int32)
    totalDS.sidList2 = np.array(totalDS.sidList2, dtype=np.int32)
    totalDS.sidList3 = np.array(totalDS.sidList3, dtype=np.int32)

    testDS = BELKA('./datasets/test.csv', protein_name=None, predict=True)

    tkn2id = {'[PAD]':-100}
    if args.modelType=='FprMLP':
        fpr_list=['rdk','Toplogical','MACCS','AtomPair','ECFP4','FCFP6','Avalon','Layered','Pattern']
        id2smiles = list(set(list(totalDS.id2cansmiles)+list(testDS.id2cansmiles)))
        id2fpr = np.hstack([get_molecule_vector(id2smiles, fpr2func[key], \
                                                convert2Mol=True, post_fn='ToList', usebar=True) for key in fpr_list])
        smi2fpr = {k:v for k,v in zip(id2smiles,id2fpr)}
        finetuneCollater = FPRCollateFunc_forBELKA(fpr_list=fpr_list, smi2fpr=smi2fpr) # , memmap=['./fpr_cache/synsmiles',len(totalDS)] , use_lmdb=False
        backbone = FprMLP_forBELKA(fprSize=16896*4, hdnSize=2048)

    elif args.modelType=='FprMLP2':
        fpr_list=['rdk','Toplogical','MACCS','AtomPair','ECFP4','FCFP6','Avalon','Layered','Pattern']
        id2smiles = list(set(list(totalDS.id2cansmiles)+list(testDS.id2cansmiles)))
        id2fpr = np.hstack([get_molecule_vector(id2smiles, fpr2func[key], \
                                                convert2Mol=True, post_fn='ToList', usebar=True) for key in fpr_list])
        smi2fpr1 = {k:v for k,v in zip(id2smiles,id2fpr)}

        with open('./fpr_cache/unimollm_smi2fpr.pkl', 'rb') as f:
            smi2fpr2 = pickle.load(f)

        with open('./fpr_cache/geminimol_smi2fpr.pkl', 'rb') as f:
            smi2fpr3 = pickle.load(f)

        with open('./fpr_cache/unimol_smi2fpr.pkl', 'rb') as f:
            smi2fpr4 = pickle.load(f)

        smi2fpr = {k:np.hstack([smi2fpr1[k],smi2fpr2[k],smi2fpr3[k],smi2fpr4[k]]) for k in smi2fpr1}
        finetuneCollater = FPRCollateFunc2_forBELKA(bertDir='./bertModel/Tokenizer_final', smi2fpr=smi2fpr, seqMaxLen=128)

        fprSize = len(smi2fpr[list(smi2fpr.keys())[0]])
        print('fprSize:', fprSize)
        backbone = FprMLP2_forBELKA(fprSize=fprSize)

    elif args.modelType.startswith('DeepFM'):
        fpr_list=['ECFP4', 'Toplogical', 'AtomPair', 'FCFP6']
        id2smiles = list(set(list(totalDS.id2cansmiles)+list(testDS.id2cansmiles)))
        id2fpr = np.hstack([get_molecule_vector(id2smiles, fpr2func[key], \
                                                convert2Mol=True, post_fn='ToList', usebar=True) for key in fpr_list])
        smi2fpr = {k:v for k,v in zip(id2smiles,id2fpr)}
        finetuneCollater = FPRCollateFunc_forBELKA(fpr_list=fpr_list, smi2fpr=smi2fpr) # , memmap=('./fpr_cache/synsmiles',len(totalDS))
        if args.modelType=='DeepFM':
            backbone = DeepFM_forBELKA()
        else:
            backbone = DeepFM2_forBELKA()

    elif args.modelType=='PseLabAttn':
        dMaxLen = min(np.max([len(i) for i in totalDS.id2smiles]),70)
        print(f'Using length {dMaxLen} for drugs...')
        finetuneCollater = DTICollateFunc_forBELKA(bertDir='./bertModel/Tokenizer_final', removeDy=True,
                                                   seqMaxLen=dMaxLen, randomSMILES=True, useFP=True)
        tkn2id = finetuneCollater.tokenizer.get_vocab()
        fHdnSizeList = [1024,256]
        backbone = PseLabAttnDPI_forBELKA(fHdnSizeList, hdnSize=64, fcSize=256, pseLabNum=16, 
                                          dropout=0.1, classNum=3, 
                                          fSize=2048)
    elif args.modelType=='GraphMLP':
        finetuneCollater = GraphCollateFunc_forBELKA(bertDir='./bertModel/Tokenizer_final')
        backbone = GraphMLP_forBELKA().cuda()
    elif args.modelType=='GraphMLP2':
        finetuneCollater = GraphCollateFunc_forBELKA(bertDir='./bertModel/Tokenizer_final', useFP=True)
        backbone = GraphMLP2_forBELKA().cuda()

    saveSteps = -1
    if args.split_type=='str':
        skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=SEED)
        trainIdx,validIdx = next(skf.split(range(len(totalDS)), y=np.sum(totalDS.Y, axis=1)))

        trainDS,validDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx)

        print(len(trainDS),len(validDS))
    elif args.split_type=='full':
        trainDS,validDS = totalDS,torch.utils.data.Subset(totalDS, [0,1,2,3,4,5,6,7,8,9,10])
        saveSteps = 1

    
    lr = float(args.lr)
    batchSize = int(args.batchSize)
    epochs = int(args.epochs)
    earlyStop = int(args.earlyStop)
    weightDecay = float(args.weightDecay)

    warmupRatio = float(args.warmupRatio)

    stepsPerEpoch = len(trainDS)//batchSize
    if ddp:
        print('world size:', args.world_size)
        stepsPerEpoch = int(stepsPerEpoch//args.world_size)

    backbone.alwaysTrain = True

    beginSteps = 1
    if args.restore is not None:
        print('Using restored weight...')
        parameters = torch.load(args.restore, map_location="cpu")
        backbone.load_state_dict(parameters['model'], strict=False)
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))
        if 0<float(args.beginSteps) and float(args.beginSteps)<1:
            beginSteps = int(float(args.beginSteps)*stepsPerEpoch)
        else:
            beginSteps = int(args.beginSteps)

    if ddp:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) #
    backbone = backbone.cuda()

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=weightDecay, eps=1e-6)

    model = HuggingfaceSeq2SeqLanguageModel2_ForClassification(backbone, optimizer=optimizer, collateFunc=finetuneCollater, 
                                                               AMP=True, DDP=ddp, 
                                                               multilabel=True)

    if args.EMA:
        print(f'using EMA... will begin at epoch {args.EMAse}')
    if args.predict_only:
        model.to_eval_mode()
        dataStream = DataLoader(validDS, batch_size=256, collate_fn=finetuneCollater, \
                                pin_memory=True, num_workers=8, prefetch_factor=16)
        with torch.no_grad():
            res = model.calculate_y_prob_by_iterator(dataStream)
        with open(args.predict_output, 'wb') as f:
            pickle.dump(res, f)
    else:
        model.train(trainDataSet=trainDS, validDataSet=validDS, batchSize=batchSize, doEvalTrain=False, beginSteps=beginSteps,
                    saveSteps=saveSteps, maxSteps=stepsPerEpoch*epochs, evalSteps=int(stepsPerEpoch), earlyStop=earlyStop, 
                    metrics="MaAUPR", report=['MaAUC','MaAUPR'], isHigherBetter=True, 
                    EMA=args.EMA, EMAse=int(args.EMAse), EMAdecay=0.999, 
                    savePath=f"./saved_models/{args.modelType}_seed{SEED}_spli{args.split_type}_ema{args.EMA}", dataLoadNumWorkers=8, pinMemory=True, ignoreIdx=tkn2id['[PAD]'], 
                    warmupSteps=int(warmupRatio*stepsPerEpoch), SEED=SEED, prefetchFactor=16)