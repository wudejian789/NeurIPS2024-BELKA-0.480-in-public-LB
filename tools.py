import torch,random,copy
import numpy as np
from isoRMSD import GetBestRMSD
from tqdm import tqdm
from rdkit import Chem
from torch.utils.data import DataLoader,Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def apply_swa(model,
              checkpoint_list: list,
              weight_list: list,
              strict: bool = True):
    """
    :param model:
    :param checkpoint_list: 要进行swa的模型路径列表
    :param weight_list: 每个模型对应的权重
    :param strict: 输入模型权重与checkpoint是否需要完全匹配
    :return:
    """
    print(weight_list)
    checkpoint_tensor_list = [torch.load(f, map_location='cuda') for f in checkpoint_list]
 
    for name, param in model.named_parameters():
        try:
            param.data = sum([ckpt['model'][name] * w for ckpt, w in zip(checkpoint_tensor_list, weight_list)])
        except KeyError:
            if strict:
                raise KeyError(f"Can't match '{name}' from checkpoint")
            else:
                print(f"Can't match '{name}' from checkpoint")
 
    return model

# 对抗训练
class FGMer():
    def __init__(self, model, emb_name='emb'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and np.any([en in name for en in self.emb_name]):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and np.any([en in name for en in self.emb_name]): 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class EMAer():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def dict_to_device(data, device):
    for k in data:
        if data[k] is None:
            continue
        if isinstance(data[k], dict):
            data[k] = dict_to_device(data[k], device)
        else:
            if hasattr(data[k],'to'):
                data[k] = data[k].to(device)
    return data

def RodriguesMatrixModel(src, dst, scale=None):
    # 计算比例关系
    if scale is None:
        scale = np.sum(np.sqrt(np.sum((dst - np.mean(dst, axis=0)) ** 2, axis=1))) / np.sum(np.sqrt(np.sum((src - np.mean(src, axis=0)) ** 2, axis=1)))
    # 计算旋转矩阵
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src = src - src_mean
    dst = dst - dst_mean
    H = np.dot(src.T, dst)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    # 计算平移矩阵
    t = dst_mean.T - scale * np.dot(R, src_mean.T)
    return R, t, scale

def BatchRodriguesMatrixModel(srcList, dstList, xyzMASK=None, scale=None):
    # src: B,L,3; dst: B,L,3; xyzMASK: B,L
    if xyzMASK is None:
        # 计算比例关系
        if scale is None:
            scale = np.sum(np.sqrt(np.sum((dstList - np.mean(dstList, axis=1,keepdims=True)) ** 2, axis=2)), axis=1) / np.sum(np.sqrt(np.sum((srcList - np.mean(srcList, axis=1,keepdims=True)) ** 2, axis=2)), axis=1)
        scale = scale[:,None,None]
        # 计算旋转矩阵
        src_mean = np.mean(srcList, axis=1, keepdims=True)
        dst_mean = np.mean(dstList, axis=1, keepdims=True)
        srcList = srcList - src_mean
        dstList = dstList - dst_mean
        H = srcList.swapaxes(-1,-2)@dstList
        U, S, Vt = np.linalg.svd(H)
        R = Vt.swapaxes(-1,-2) @ U.swapaxes(-1,-2)
        
        isUsed = np.linalg.det(R) < 0
        Vt[isUsed, 2, :] *= -1
        R[isUsed] = Vt[isUsed].swapaxes(-1,-2) @ U[isUsed].swapaxes(-1,-2)
        # 计算平移矩阵
        t = dst_mean.swapaxes(-1,-2) - scale * (R@src_mean.swapaxes(-1,-2))
        return R, t.swapaxes(-1,-2), scale
    else:
        RArr,tArr,scaleArr = [],[],[]
        for src,dst,mask in zip(srcList,dstList,xyzMASK):
            src,dst = src[mask],dst[mask]
            # 计算比例关系
            if scale is None:
                s = np.sum(np.sqrt(np.sum((dst - np.mean(dst, axis=0)) ** 2, axis=1))) / np.sum(np.sqrt(np.sum((src - np.mean(src, axis=0)) ** 2, axis=1)))
            else:
                s = 1.0
            # 计算旋转矩阵
            src_mean = np.mean(src, axis=0)
            dst_mean = np.mean(dst, axis=0)
            src = src - src_mean
            dst = dst - dst_mean
            H = np.dot(src.T, dst)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = np.dot(Vt.T, U.T)
            # 计算平移矩阵
            t = dst_mean.T - s * np.dot(R, src_mean.T)
            RArr.append(R)
            tArr.append(t)
            scaleArr.append(s)
        return np.array(RArr, dtype=np.float32),np.array(tArr, dtype=np.float32),np.array(scaleArr, dtype=np.float32)

def compute_RMSD(model, collater, dataset):
    rmsdList = []
    alignedrmsdList = []
    mean,std = collater.normVec
    for item in tqdm(dataset):
        i = dict_to_device(collater([item]), device=next(model.model.parameters()).device)
        if 'anchorNum' in i:
            anchorNum = int(i['anchorNum'])
            with torch.no_grad():
                tmp = model.model(i, predict=True)
            res = {'y_pre':tmp['y_logit'].detach().cpu().data.numpy()[0,anchorNum:], 
                   'anchorXYZ_pre':tmp['y_logit'].detach().cpu().data.numpy()[0,:anchorNum], 
                   'anchor_pre':tmp['anchor_logit'].detach().cpu().data.numpy()[0]}
            
            # align anchor_pre and anchor
            src = res['anchorXYZ_pre']
            isAnchor = (-res['anchor_pre']).argsort().argsort()<anchorNum
            anchorXYZ = (item['cy'][:len(res['anchor_pre'])][isAnchor[:len(item['cy'])]]-mean)/std
            dst = anchorXYZ
            R,t,scale = RodriguesMatrixModel(src, dst, scale=1)
            res['y_pre'] = ((scale*res['y_pre'])@R.T+t)
        else:
            with torch.no_grad():
                tmp = model.calculate_y_pre(i)
            res = {'y_pre':tmp['y_pre'].detach().cpu().data.numpy()[0]}

        xyzArr_,xyzArr_pre_ = i['y'].detach().cpu().data.numpy().reshape(-1,3),res['y_pre']
        if 'anchorNum' in i:
            xyzArr_ = xyzArr_[anchorNum:]
        xyzArr = xyzArr_*std + mean
        xyzArr_pre = (xyzArr_pre_*std + mean)
        
        if collater.isLocalEncode:
            xyzArr = xyzDecode(xyzArr, stType=collater.stType)
            if collater.stType=='XYZ':
                xyzArr_pre[0,:] = 0
                xyzArr_pre[1,1:] = 0
                xyzArr_pre[2,2] = 0
            else:
                xyzArr_pre[0,:] = 0
                xyzArr_pre[1,:-1] = 0
                xyzArr_pre[2,0] = 0
            xyzArr_pre = xyzDecode(xyzArr_pre, stType=collater.stType)
        if np.isnan(xyzArr_pre).any():
            continue
    #     break
        if isinstance(xyzArr, int) or isinstance(xyzArr_pre, int): 
            rmsdList.append(-1)
            continue
        
        if item['mol'].HasProp('_smilesAtomOutputOrder') and collater.train:
            smi2molOrder = [int(j) for j in item['mol'].GetProp('_smilesAtomOutputOrder')[1:-2].split(',')]
        else:
            smi2molOrder = item['smi2molOrder']
        atomIdx = np.argsort(smi2molOrder)
        xyzArr = xyzArr[atomIdx]
        xyzArr_pre = xyzArr_pre[atomIdx]
        
        rmsd = np.sqrt(np.mean(np.sum((xyzArr_pre - xyzArr)**2, axis=-1)))
        rmsdList.append(rmsd)
        
        mol = copy.deepcopy(item['mol'])
        mol.RemoveAllConformers()
        with open('structure_prediction/test_pre.mol', 'w') as f:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(len(xyzArr_pre)):
                conf.SetAtomPosition(i, (xyzArr_pre[i]).tolist())
            mol.AddConformer(conf)
            s = Chem.MolToMolBlock(mol)
            f.write(s)
        mol2 = Chem.MolFromMolFile('./structure_prediction/test_pre.mol')
        
        mol = copy.deepcopy(item['mol']) # Chem.MolFromSmiles(x['smiles'])
        mol.RemoveAllConformers()
        with open('structure_prediction/test.mol', 'w') as f:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(len(xyzArr)):
                conf.SetAtomPosition(i, (xyzArr[i]).tolist())
            mol.AddConformer(conf)
            # AllChem.EmbedMolecule(mol, confId=0)
            s = Chem.MolToMolBlock(mol)
            f.write(s)
    #     break
        mol1 = Chem.MolFromMolFile('./structure_prediction/test.mol')
        alignedrmsd = GetBestRMSD(mol1,mol2)[0]
        alignedrmsdList.append(alignedrmsd)
        with open('structure_prediction/aligned_test_pre.mol', 'w') as f:
            f.write(Chem.MolToMolBlock(mol2))
        with open('structure_prediction/aligned_test.mol', 'w') as f:
            f.write(Chem.MolToMolBlock(mol1))
    rmsdList = np.array(rmsdList, dtype=np.float32)
    alignedrmsdList = np.array(alignedrmsdList, dtype=np.float32)
    return alignedrmsdList.mean(), rmsdList.mean()

def batch_compute_RMSD(model, collater, dataset, batchSize=16, outFile=False, num_workers=4, prefetch_factor=8):
    rmsdList = []
    alignedrmsdList = []
    mean,std = collater.normVec
    for batch in tqdm(DataLoader(dataset, batch_size=batchSize, pin_memory=True, collate_fn=collater, num_workers=num_workers, prefetch_factor=prefetch_factor)):
        batch = dict_to_device(batch, device='cuda')
        xyz_mask = batch['xyz_mask'].detach().cpu().data.numpy()
        B = len(xyz_mask)
        xyzArr_ = batch['y'].detach().cpu().data.numpy()
        if 'anchorNum' in batch:
            anchorNum = int(batch['anchorNum'])
            xyzArr_ = xyzArr_[:,anchorNum:]
            xyz_mask = xyz_mask[:,anchorNum:]
            
            with torch.no_grad():
                tmp = model.model(batch, predict=True)
            res = {'y_pre':tmp['y_logit'].detach().cpu().data.numpy()[:,anchorNum:], 
                   'anchorXYZ_pre':tmp['y_logit'].detach().cpu().data.numpy()[:,:anchorNum], 
                   'anchor_pre':tmp['anchor_logit'].detach().cpu().data.numpy()}
            
            # align anchor_pre and anchor
            src = res['anchorXYZ_pre']
            isAnchor = (-res['anchor_pre']).argsort(axis=1).argsort(axis=1)<anchorNum
            # print(batch['cy'].shape, isAnchor.shape)
            anchorXYZ = batch['cy'][isAnchor,:].reshape(-1,anchorNum,3).detach().cpu().data.numpy()
            dst = anchorXYZ
            R,t,scale = BatchRodriguesMatrixModel(src, dst, scale=np.ones(B))
            res['y_pre'] = ((scale*res['y_pre'])@R.swapaxes(-1,-2)+t)
        else:
            with torch.no_grad():
                tmp = model.calculate_y_pre(batch)
            res = {'y_pre':tmp['y_pre'].detach().cpu().data.numpy()}
        # break
        xyzArr_pre_ = res['y_pre']
            
        xyzArr = xyzArr_*std + mean
        xyzArr_pre = (xyzArr_pre_*std + mean)
            
        if collater.isLocalEncode:
            for i in range(B):
                xyzArr[idx] = xyzDecode(xyzArr[idx], stType=stType)
                if stType=='XYZ':
                    xyzArr_pre[i,0,:] = 0
                    xyzArr_pre[i,1,1:] = 0
                    xyzArr_pre[i,2,2] = 0
                else:
                    xyzArr_pre[i,0,:] = 0
                    xyzArr_pre[i,1,:-1] = 0
                    xyzArr_pre[i,2,0] = 0
                xyzArr_pre[idx] = xyzDecode(xyzArr_pre[idx], stType=stType)
        if isinstance(xyzArr, int) or isinstance(xyzArr_pre, int): 
            rmsdList.append(np.nan)
            alignedrmsdList.append(np.nan)
            continue
        
        # xyzArr_pre, xyzArr: B,L,3
        xyzArr[~xyz_mask] = xyzArr_pre[~xyz_mask] = 0
        
        rmsd = np.sqrt(np.sum(np.sum((xyzArr_pre - xyzArr)**2, axis=-1), axis=1) / xyz_mask.sum(axis=1))
        rmsdList += rmsd.tolist()
        
        for i in range(B):
            xyz,xyz_pre = xyzArr[i][xyz_mask[i]],xyzArr_pre[i][xyz_mask[i]]
            source = collater.tokenizer.decode(batch['batch']['input_ids'][i])
            mol = Chem.MolFromSmiles(source[source.find(' '):source.find('[SEP]')].replace(' ',''))
            
            mol.RemoveAllConformers()
            if outFile:
                f = open('structure_prediction/test_pre.mol', 'w')
            else:
                f = None

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(len(xyz_pre)):
                conf.SetAtomPosition(i, (xyz_pre[i]).tolist())
            mol.AddConformer(conf)
            if f is not None:
                # AllChem.EmbedMolecule(mol, confId=0)
                s = Chem.MolToMolBlock(mol)
                f.write(s)
                f.close()
                mol2 = Chem.MolFromMolFile('./structure_prediction/test_pre.mol')
            else:
                mol2 = copy.deepcopy(mol)

            mol.RemoveAllConformers()
            if outFile:
                f = open('structure_prediction/test.mol', 'w')
            else:
                f = None

            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(len(xyz)):
                conf.SetAtomPosition(i, (xyz[i]).tolist())
            mol.AddConformer(conf)
            if f is not None:
                # AllChem.EmbedMolecule(mol, confId=0)
                s = Chem.MolToMolBlock(mol)
                f.write(s)
                f.close()
                mol1 = Chem.MolFromMolFile('./structure_prediction/test.mol')
            else:
                mol1 = copy.deepcopy(mol)

            alignedrmsd = GetBestRMSD(mol1,mol2)[0]
            alignedrmsdList.append(alignedrmsd)

            if outFile:
                with open('structure_prediction/aligned_test_pre.mol', 'w') as f:
                    f.write(Chem.MolToMolBlock(mol2))
                with open('structure_prediction/aligned_test.mol', 'w') as f:
                    f.write(Chem.MolToMolBlock(mol1))

    rmsdList = np.array(rmsdList, dtype=np.float32)
    alignedrmsdList = np.array(alignedrmsdList, dtype=np.float32)

    return alignedrmsdList.mean(), rmsdList.mean()

def get_Ri_inv_true_based_on_adjMat(xyzArr, adjArr, aWtArr=None):
    # xyzArr: B,L,3; adjArr: B,L,L; aWtArr: B,L
    B,L,_ = xyzArr.shape
    adjArr_sec_ord = adjArr@adjArr # B,L,L

    anchorArr = torch.zeros((B,L,2,3), dtype=xyzArr.dtype, device=adjArr.device) # B,L,2,3
    for i in range(B):
        for j in range(L):
            if adjArr[i,j].sum()>2:
                isUsed = (adjArr[i,j]>0).bool()
                isUsed[j] = False
                anchorArr[i,j] = xyzArr[i,isUsed][:2] # 2,3
                awt = aWtArr[i,isUsed][:2]
                if awt[0]<awt[1]:
                    anchorArr[i,j] = anchorArr[i,j,[1,0]]
            else:
                isUsed = (adjArr_sec_ord[i,j]>0).bool()
                isUsed[j] = False
                if isUsed.sum()>1:
                    anchorArr[i,j] = xyzArr[i,isUsed][:2] # 2,3
                    awt = aWtArr[i,isUsed][:2]
                    if awt[0]<awt[1]:
                        anchorArr[i,j] = anchorArr[i,j,[1,0]]
                else:
                    pass
    ex = anchorArr[:,:,0] - xyzArr
    ex = ex / torch.sqrt(torch.sum(ex**2, dim=-1, keepdims=True)) # B,L,3

    et = anchorArr[:,:,1] - xyzArr

    ey = et - torch.sum(et*ex, dim=-1, keepdims=True)*ex
    ey = ey / torch.sqrt(torch.sum(ey**2, dim=-1, keepdims=True)) # B,L,3

    ez_x = ex[:,:,1]*ey[:,:,2] - ex[:,:,2]*ey[:,:,1]
    ez_y = ex[:,:,2]*ey[:,:,0] - ex[:,:,0]*ey[:,:,2]
    ez_z = ex[:,:,0]*ey[:,:,1] - ex[:,:,1]*ey[:,:,0]
    ez = torch.cat([ez_x[:,None,:],ez_y[:,None,:],ez_z[:,None,:]], dim=1).transpose(-1,-2) # B,L-2,3

    Ri = torch.cat([ex[:,:,None],ey[:,:,None],ez[:,:,None]], dim=2) # B,L,3,3
    return Ri.transpose(-1,-2)

def get_Ri_inv_true(xyzArr):
    # input:  B,L,3
    # output: B,L-2,3,3
    ex = xyzArr[:,1:-1] - xyzArr[:,:-2]
    ex = ex / torch.sqrt(torch.sum(ex**2, dim=-1, keepdims=True)) # B,L-2,3

    et = xyzArr[:,2:] - xyzArr[:,:-2]

    ey = et - torch.sum(et*ex, dim=-1, keepdims=True)*ex
    ey = ey / torch.sqrt(torch.sum(ey**2, dim=-1, keepdims=True)) # B,L-2,3

    ez_x = ex[:,:,1]*ey[:,:,2] - ex[:,:,2]*ey[:,:,1]
    ez_y = ex[:,:,2]*ey[:,:,0] - ex[:,:,0]*ey[:,:,2]
    ez_z = ex[:,:,0]*ey[:,:,1] - ex[:,:,1]*ey[:,:,0]
    ez = torch.cat([ez_x[:,None,:],ez_y[:,None,:],ez_z[:,None,:]], dim=1).transpose(-1,-2) # B,L-2,3

    Ri = torch.cat([ex[:,:,None],ey[:,:,None],ez[:,:,None]], dim=2) # B,L-2,3,3
    return Ri.transpose(-1,-2)

def get_Ri_inv_true_based_on_anchor(xyzArr, anchorArr):
    # xyzArr: B,L,3
    # anchorArr: B,an,3
    ex = anchorArr[:,[-1]] - xyzArr
    ex = ex / torch.sqrt(torch.sum(ex**2, dim=-1, keepdims=True)) # B,L,3

    et = anchorArr[:,[-2]] - xyzArr

    ey = et - torch.sum(et*ex, dim=-1, keepdims=True)*ex
    ey = ey / torch.sqrt(torch.sum(ey**2, dim=-1, keepdims=True)) # B,L,3

    ez_x = ex[:,:,1]*ey[:,:,2] - ex[:,:,2]*ey[:,:,1]
    ez_y = ex[:,:,2]*ey[:,:,0] - ex[:,:,0]*ey[:,:,2]
    ez_z = ex[:,:,0]*ey[:,:,1] - ex[:,:,1]*ey[:,:,0]
    ez = torch.cat([ez_x[:,None,:],ez_y[:,None,:],ez_z[:,None,:]], dim=1).transpose(-1,-2) # B,L-2,3

    Ri = torch.cat([ex[:,:,None],ey[:,:,None],ez[:,:,None]], dim=2) # B,L,3,3
    return Ri.transpose(-1,-2)

def xyzEncode(xyzArr, stType='DDD'):
    assert stType in ['DDD','AAD','XYZ']
    # judge if there are three continuous points sharing a same line
    pass

    # calculate the structure information features
    res = np.zeros((len(xyzArr), 3), dtype=np.float32)
    
    if stType=='XYZ':

        if len(xyzArr)>1:
            res[1,0] = np.sqrt(np.sum((xyzArr[1] - xyzArr[0])**2, axis=-1))
        if len(xyzArr)>2:
            ex = xyzArr[1]-xyzArr[0]
            ex = ex / np.sqrt(np.sum(ex**2, axis=-1, keepdims=True))

            et = xyzArr[2]-xyzArr[0]

            ey = et - np.sum(et*ex, axis=-1)*ex
            ey = ey / np.sqrt(np.sum(ey**2, axis=-1, keepdims=True))

            res[2,0],res[2,1] = np.sum(et*ex,axis=-1),np.sum(et*ey,axis=-1)
        if len(xyzArr)>3:
            ex = xyzArr[1:-2] - xyzArr[:-3]
            ex = ex / np.sqrt(np.sum(ex**2, axis=-1, keepdims=True))

            et = xyzArr[2:-1] - xyzArr[:-3]

            ey = et - np.sum(et*ex, axis=-1, keepdims=True)*ex
            ey = ey / np.sqrt(np.sum(ey**2, axis=-1, keepdims=True))

            ez_x = ex[:,1]*ey[:,2] - ex[:,2]*ey[:,1]
            ez_y = ex[:,2]*ey[:,0] - ex[:,0]*ey[:,2]
            ez_z = ex[:,0]*ey[:,1] - ex[:,1]*ey[:,0]
            ez = np.vstack([ez_x,ez_y,ez_z]).T

            p1p4 = xyzArr[3:] - xyzArr[:-3]
            res[3:,0],res[3:,1],res[3:,2] = np.sum(p1p4*ex,axis=-1),np.sum(p1p4*ey,axis=-1),np.sum(p1p4*ez,axis=-1)
    else:

        if len(xyzArr)>1:
            p1_p2 = xyzArr[1:] - xyzArr[:-1]
            d1 = np.sqrt((p1_p2**2).sum(axis=1))

            res[1:,2] = d1

        if stType=='DDD':
            if len(xyzArr)>2:
                p1_p3 = xyzArr[2:] - xyzArr[:-2]
                d2 = np.sqrt((p1_p3**2).sum(axis=1))

                res[2:,1] = d2

            if len(xyzArr)>3:
                p1_p4 = xyzArr[3:] - xyzArr[:-3]
                d3 = np.sqrt((p1_p4**2).sum(axis=1))
                res[3:,0] = d3

                p1_p2 = p1_p2[:-2]
                p1_p3 = p1_p3[:-1]

                n_p1p2p3_x = p1_p2[:,1]*p1_p3[:,2] - p1_p2[:,2]*p1_p3[:,1]
                n_p1p2p3_y = p1_p2[:,2]*p1_p3[:,0] - p1_p2[:,0]*p1_p3[:,2]
                n_p1p2p3_z = p1_p2[:,0]*p1_p3[:,1] - p1_p2[:,1]*p1_p3[:,0]
                n_p1p2p3 = np.vstack([n_p1p2p3_x,n_p1p2p3_y,n_p1p2p3_z]).T
                # n_p1p2p3 = np.cross(p1_p2, p1_p3) # this build-in function is too slow

                l2 = (p1_p4**2).sum(axis=1) * (n_p1p2p3**2).sum(axis=1) # if l2==0, indicate p1p2p3 share a line=>|n_p1p2p3|=0
                theta = np.pi/2 - np.arccos(np.clip((p1_p4*n_p1p2p3).sum(axis=1) / np.sqrt(l2), -1.0,1.0))
                
                res[3:][theta<0] *= -1

        elif stType=='AAD':
            if len(xyzArr)>2:
                p2_p1 = -p1_p2[:-1]
                p2_p3 = p1_p2[1:]
                
                theta = np.arccos(np.clip((p2_p1*p2_p3).sum(axis=1) / np.sqrt((p2_p1**2).sum(axis=1) * (p2_p3**2).sum(axis=1)), -1.0,1.0))

                res[2:,1] = theta

            if len(xyzArr)>3:
                p2_p1 = p2_p1[:-1]
                p2_p3 = p2_p3[:-1]
                p3_p4 = p1_p2[2:]

                n_p2p1p3_x = p2_p1[:,1]*p2_p3[:,2] - p2_p1[:,2]*p2_p3[:,1]
                n_p2p1p3_y = p2_p1[:,2]*p2_p3[:,0] - p2_p1[:,0]*p2_p3[:,2]
                n_p2p1p3_z = p2_p1[:,0]*p2_p3[:,1] - p2_p1[:,1]*p2_p3[:,0]
                n_p2p1p3 = np.vstack([n_p2p1p3_x,n_p2p1p3_y,n_p2p1p3_z]).T
                # n_p2p1p3 = np.cross(p1_p2, p1_p3) # this build-in function is too slow

                p3_p2 = -p2_p3
                n_p3p4p2_x = p3_p4[:,1]*p3_p2[:,2] - p3_p4[:,2]*p3_p2[:,1]
                n_p3p4p2_y = p3_p4[:,2]*p3_p2[:,0] - p3_p4[:,0]*p3_p2[:,2]
                n_p3p4p2_z = p3_p4[:,0]*p3_p2[:,1] - p3_p4[:,1]*p3_p2[:,0]
                n_p3p4p2 = np.vstack([n_p3p4p2_x,n_p3p4p2_y,n_p3p4p2_z]).T

                l2 = (n_p2p1p3**2).sum(axis=1) * (n_p3p4p2**2).sum(axis=1)
                theta = np.pi - np.arccos(np.clip((n_p2p1p3*n_p3p4p2).sum(axis=1) / np.sqrt(l2), -1.0,1.0)) # if l2==2, indicate p1p2p3 or p2p3p4 share a line => |n_p1p2p3|or|n_p2p3p4|=0
                
                theta[l2==0] = 0

                res[3:,0] = theta
                
                l2 = (p3_p4**2).sum(axis=1) * (n_p2p1p3**2).sum(axis=1) # if l2==0, indicate p1p2p3 share a line=>|n_p1p2p3|=0
                theta = np.pi/2 - np.arccos(np.clip((p3_p4*n_p2p1p3).sum(axis=1) / np.sqrt(l2), -1.0,1.0))
                res[3:,0][theta<-1e-6] *= -1
            
    return res

def xyzDecode(tmp, stType='DDD', eps=1e-3, sIdx=0, xyzArr=None):
    assert stType in ['DDD','AAD','XYZ']
    if xyzArr is None:
        xyzArr = np.zeros((len(tmp),3), dtype=np.float32)
    
    for idx in range(sIdx,len(tmp)):
        if idx==0:
            # put the first atom at the origin 
            continue
        elif idx==1:
            # put the second atom at the x-axis
            if stType=='XYZ':
                xyzArr[idx][0] = tmp[idx][0]
            else:
                xyzArr[idx][0] = tmp[idx][2]
        elif idx==2:
            # put the third atom at the x-y plane
            if stType=='XYZ':
                ex = xyzArr[idx-1] - xyzArr[idx-2]
                ex = ex / np.sqrt(np.sum(ex**2, axis=-1))

                ey = np.array([ex[1],-ex[0],0], dtype=ex.dtype)
                ey = ey / np.sqrt(np.sum(ey**2, axis=-1))

                assert tmp[idx][2]==0
                xyzArr[idx] = tmp[idx][0]*ex + tmp[idx][1]*ey
            elif stType=='DDD':
                xyzArr[idx][0] = (xyzArr[idx-1][0]**2 + tmp[idx][1]**2 - tmp[idx][2]**2) / (2*xyzArr[idx-1][0])
                xyzArr[idx][1] = np.sqrt(np.abs(tmp[idx][1]**2 - xyzArr[idx][0]**2))
            elif stType=='AAD':
                xyzArr[idx][0] = xyzArr[idx-1][0] - tmp[idx][2]*np.cos(tmp[idx][1])
                xyzArr[idx][1] = tmp[idx][2]*np.sin(tmp[idx][1])
        else:
            if stType=='XYZ':
                ex = xyzArr[idx-2] - xyzArr[idx-3]
                ex = ex / np.sqrt(np.sum(ex**2, axis=-1))

                et = xyzArr[idx-1] - xyzArr[idx-3]

                ey = et - np.sum(et*ex, axis=-1)*ex
                if np.sqrt(np.sum(ey**2))<eps:
                    print('three points (XYZ) are in a same line, ignore this samples')
                    return -1
                ey = ey / np.sqrt(np.sum(ey**2, axis=-1))

                ez_x = ex[1]*ey[2] - ex[2]*ey[1]
                ez_y = ex[2]*ey[0] - ex[0]*ey[2]
                ez_z = ex[0]*ey[1] - ex[1]*ey[0]
                ez = np.array([ez_x,ez_y,ez_z], dtype=xyzArr.dtype)

                xyzArr[idx] = xyzArr[idx-3] + tmp[idx][0]*ex + tmp[idx][1]*ey + tmp[idx][2]*ez
            elif stType=='DDD':
                r1,r2,r3 = np.abs(tmp[idx-1][1]),np.abs(tmp[idx-1][2]),np.abs(tmp[idx-2][2])
                if np.abs(r1-(r2+r3))<eps or np.abs(r2-(r1+r3))<eps or np.abs(r3-(r1+r2))<eps:
                    print('three points (DDD) are in a same line, ignore this samples')
                    return -1
                r1,r2,r3 = np.abs(tmp[idx][1]),np.abs(tmp[idx][2]),np.abs(tmp[idx-1][2])
                if np.abs(r1-(r2+r3))<eps or np.abs(r2-(r1+r3))<eps or np.abs(r3-(r1+r2))<eps:
                    print('three points (DDD) are in a same line, ignore this samples')
                    return -1
                
                # put the next atoms according to (d1,d2,d3)
                r1,r2,r3 = tmp[idx]
                P1P2 = xyzArr[idx-2] - xyzArr[idx-3]
                d = np.sqrt(np.sum(P1P2**2))
                ex = P1P2 / d
                P1P3 = xyzArr[idx-1] - xyzArr[idx-3]
                i = (ex*P1P3).sum()
                
                EP3 = P1P3 - i*ex
                d_EP3 = np.sqrt(np.sum(EP3**2))
                ey = EP3 / d_EP3
                j = (ey*P1P3).sum()
                
                ez = np.cross(ex, ey)
                
                x_ = (r1**2 + d**2 - r2**2) / (2*d)
                y_ = (r1**2 - r3**2 - 2*i*x_ + i**2 + j**2) / (2*j)

                z_ = np.sqrt(np.abs(r1**2 - x_**2 - y_**2))
                if (tmp[idx]>0).sum()<(tmp[idx]<0).sum(): z_ = -z_

                xyz = xyzArr[idx-3] + x_*ex + y_*ey + z_*ez
                xyzArr[idx] = xyz
            elif stType=='AAD':
                if np.abs(tmp[idx-1][1]-np.pi)<eps or np.abs(tmp[idx][1]-np.pi)<eps or np.abs(tmp[idx-1][1])<eps or np.abs(tmp[idx][1])<eps:
                    print('three points (AAD) are in a same line, ignore this samples')
                    return -1
                
                # put the next atoms according to (beta,alpha,d)
                beta,alpha,d = tmp[idx]
                
                P2P3 = xyzArr[idx-1] - xyzArr[idx-2]
                d_P2P3 = np.sqrt(np.sum(P2P3**2))
                ex = P2P3 / d_P2P3
                
                P1P2 = xyzArr[idx-2] - xyzArr[idx-3]
                P1E = (P1P2*ex).sum() * ex
                EP2 = P1P2 - P1E
                ey = EP2 / np.sqrt(np.sum(EP2**2))
                ez = np.cross(ex,ey)
                
                d_P2D = d_P2P3 - d*np.cos(alpha)
                d_P4D = d*np.sin(alpha)
                d_CD = d_P4D*np.cos(beta)
                d_P4C = d_P4D*np.sin(beta)
                
                d_P1E = np.sqrt(np.sum(P1E**2))
                d_P2E = np.sqrt(np.sum(EP2**2))
                
                if beta<0: d_P4D *= -1
                
                theta = np.pi/2 - np.arccos(np.sum(P1P2*P2P3) / np.sqrt(np.sum(P1P2**2)*np.sum(P2P3**2)))
                if theta<0:
                    d_P1E *= -1
                
                xyz = xyzArr[idx-3] + (d_P1E+d_P2D)*ex + (d_P2E-d_CD)*ey + d_P4C*ez
                xyzArr[idx] = xyz
    return xyzArr

def AADL2DDDL(aadl):
    # aadl: B × n × 3
    d_1 = aadl[:,1:,[2]] # B × (n-1) × 1
    d_2 = torch.sqrt(torch.abs(aadl[:,2:,[2]]**2 + aadl[:,:-2,[2]]**2 - 2*aadl[:,2:,[2]]*aadl[:,:-2,[2]]*torch.cos(aadl[:,2:,[1]]))) # B × (n-2) × 1

    a = aadl[:,3:,[2]]*torch.sin(aadl[:,3:,[1]])
    b = aadl[:,1:-2,[2]]*torch.sin(aadl[:,2:-1,[1]])
    c = aadl[:,2:-1,[2]] - aadl[:,3:,[2]]*torch.cos(aadl[:,3:,[1]]) - aadl[:,1:-2,[2]]*torch.cos(aadl[:,2:-1,[1]])

    d = a*torch.sin(aadl[:,3:,[0]])
    e = torch.sqrt((b-a*torch.cos(aadl[:,3:,[1]]))**2 + c**2)
 
    d_3 = torch.sqrt(d**2 + e**2) # B × (n-3) × 1

    dddl = torch.zeros_like(aadl, dtype=aadl.dtype)

    dddl[:,1:,[2]] += d_1
    dddl[:,2:,[1]] += d_2
    dddl[:,3:,[0]] += d_3

    dddl[:,3:] *= aadl[:,3:,[0]] / (torch.abs(aadl[:,3:,[0]])+1e-6)

    return dddl

def torch_xyzEncode(xyzArr, stType='DDD'):
    assert stType in ['DDD','AAD','XYZ']
    # judge if there are three continuous points sharing a same line
    pass

    # calculate the structure information features
    res = torch.zeros(xyzArr.shape, dtype=xyzArr.dtype, device=xyzArr.device)
    if stType=='XYZ':

        if xyzArr.shape[1]>1:
            res[:,1,0] = torch.sqrt(torch.sum((xyzArr[:,1] - xyzArr[:,0])**2, dim=-1))
        if xyzArr.shape[1]>2:
            ex = xyzArr[:,1]-xyzArr[:,0]
            ex = ex / torch.sqrt(torch.sum(ex**2, dim=-1, keepdims=True))

            et = xyzArr[:,2]-xyzArr[:,0]

            ey = et - torch.sum(et*ex, dim=-1, keepdims=True)*ex
            ey = ey / torch.sqrt(torch.sum(ey**2, dim=-1, keepdims=True))

            res[:,2,0],res[:,2,1] = torch.sum(et*ex,dim=-1),torch.sum(et*ey,dim=-1)
        if xyzArr.shape[1]>3:
            ex = xyzArr[:,1:-2] - xyzArr[:,:-3]
            ex = ex / torch.sqrt(torch.sum(ex**2, dim=-1, keepdims=True))

            et = xyzArr[:,2:-1] - xyzArr[:,:-3]

            ey = et - torch.sum(et*ex, dim=-1, keepdims=True)*ex
            ey = ey / torch.sqrt(torch.sum(ey**2, dim=-1, keepdims=True))

            ez_x = ex[:,:,1]*ey[:,:,2] - ex[:,:,2]*ey[:,:,1]
            ez_y = ex[:,:,2]*ey[:,:,0] - ex[:,:,0]*ey[:,:,2]
            ez_z = ex[:,:,0]*ey[:,:,1] - ex[:,:,1]*ey[:,:,0]
            ez = torch.cat([ez_x[:,None,:],ez_y[:,None,:],ez_z[:,None,:]], dim=1).transpose(-1,-2)

            p1p4 = xyzArr[:,3:] - xyzArr[:,:-3]
            res[:,3:,0],res[:,3:,1],res[:,3:,2] = torch.sum(p1p4*ex,dim=-1),torch.sum(p1p4*ey,dim=-1),torch.sum(p1p4*ez,dim=-1)
    else:

        if xyzArr.shape[1]>1:
            p1_p2 = xyzArr[:,1:] - xyzArr[:,:-1]
            d1 = torch.sqrt((p1_p2**2).sum(dim=-1))

            res[:,1:,2] += d1

        if stType=='DDD':
            print(1/0)
            # if len(xyzArr)>2:
            #     p1_p3 = xyzArr[2:] - xyzArr[:-2]
            #     d2 = torch.sqrt((p1_p3**2).sum(dim=1))

            #     res[2:,1] = d2

            # if len(xyzArr)>3:
            #     p1_p4 = xyzArr[3:] - xyzArr[:-3]
            #     d3 = torch.sqrt((p1_p4**2).sum(dim=1))
            #     res[3:,0] = d3

            #     p1_p2 = p1_p2[:-2]
            #     p1_p3 = p1_p3[:-1]

            #     n_p1p2p3_x = p1_p2[:,1]*p1_p3[:,2] - p1_p2[:,2]*p1_p3[:,1]
            #     n_p1p2p3_y = p1_p2[:,2]*p1_p3[:,0] - p1_p2[:,0]*p1_p3[:,2]
            #     n_p1p2p3_z = p1_p2[:,0]*p1_p3[:,1] - p1_p2[:,1]*p1_p3[:,0]
            #     n_p1p2p3 = torch.vstack([n_p1p2p3_x,n_p1p2p3_y,n_p1p2p3_z]).T

            #     l2 = (p1_p4**2).sum(dim=1) * (n_p1p2p3**2).sum(dim=1) # if l2==0, indicate p1p2p3 share a line=>|n_p1p2p3|=0
            #     theta = torch.pi/2 - torch.arccos(torch.clip((p1_p4*n_p1p2p3).sum(dim=1) / torch.sqrt(l2+1e-6), -1.0,1.0))
                
            #     res[3:][theta<0] *= -1

        elif stType=='AAD':
            if xyzArr.shape[1]>2:
                p2_p1 = -p1_p2[:,:-1]
                p2_p3 = p1_p2[:,1:]
                
                theta = torch.arccos(torch.clip((p2_p1*p2_p3).sum(dim=-1) / torch.sqrt((p2_p1**2).sum(dim=-1) * (p2_p3**2).sum(dim=-1) + 1e-6), -1.0,1.0))

                res[:,2:,1] += theta

            if xyzArr.shape[1]>3:
                p2_p1 = p2_p1[:,:-1]
                p2_p3 = p2_p3[:,:-1]
                p3_p4 = p1_p2[:,2:]

                n_p2p1p3_x = p2_p1[:,:,1]*p2_p3[:,:,2] - p2_p1[:,:,2]*p2_p3[:,:,1]
                n_p2p1p3_y = p2_p1[:,:,2]*p2_p3[:,:,0] - p2_p1[:,:,0]*p2_p3[:,:,2]
                n_p2p1p3_z = p2_p1[:,:,0]*p2_p3[:,:,1] - p2_p1[:,:,1]*p2_p3[:,:,0]
                n_p2p1p3 = torch.cat([n_p2p1p3_x[:,None,:],n_p2p1p3_y[:,None,:],n_p2p1p3_z[:,None,:]], dim=1).transpose(-1,-2)

                p3_p2 = -p2_p3
                n_p3p4p2_x = p3_p4[:,:,1]*p3_p2[:,:,2] - p3_p4[:,:,2]*p3_p2[:,:,1]
                n_p3p4p2_y = p3_p4[:,:,2]*p3_p2[:,:,0] - p3_p4[:,:,0]*p3_p2[:,:,2]
                n_p3p4p2_z = p3_p4[:,:,0]*p3_p2[:,:,1] - p3_p4[:,:,1]*p3_p2[:,:,0]
                n_p3p4p2 = torch.cat([n_p3p4p2_x[:,None,:],n_p3p4p2_y[:,None,:],n_p3p4p2_z[:,None,:]], dim=1).transpose(-1,-2)

                l2 = (n_p2p1p3**2).sum(dim=-1) * (n_p3p4p2**2).sum(dim=-1)
                theta = torch.pi - torch.arccos(torch.clip((n_p2p1p3*n_p3p4p2).sum(dim=-1) / torch.sqrt(l2+1e-6), -1.0,1.0)) # if l2==2, indicate p1p2p3 or p2p3p4 share a line => |n_p1p2p3|or|n_p2p3p4|=0
                
                theta[l2==0] = 0

                res[:,3:,0] += theta
                
                l2 = (p3_p4**2).sum(dim=-1) * (n_p2p1p3**2).sum(dim=-1) # if l2==0, indicate p1p2p3 share a line=>|n_p1p2p3|=0
                theta = torch.pi/2 - torch.arccos(torch.clip((p3_p4*n_p2p1p3).sum(dim=-1) / torch.sqrt(l2+1e-6), -1.0,1.0))
                res[:,3:,0][theta<-1e-6] *= -1

    # if torch.isnan(res).any():
    #     print(1/0)
            
    return res