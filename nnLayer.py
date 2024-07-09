from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict

class FFN_GatedLinearUnit(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='GLU'):
        super(FFN_GatedLinearUnit, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.WU = nn.Sequential(nn.Linear(feaSize, feaSize*4, bias=False),
                                nn.SiLU())
        self.WV = nn.Linear(feaSize,feaSize*4, bias=False)
        self.WO = nn.Linear(feaSize*4, feaSize, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize
        ffnx = self.WO(self.WU(z)*self.WV(z))
        return self.layerNorm2(z+self.dropout(ffnx))

class TextEmbedding(nn.Module):
    def __init__(self, embedding, tknDropout=0.3, embDropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=tknDropout)
        self.dropout2 = nn.Dropout(p=embDropout)
    def forward(self, x):
        # x: batchSize × seqLen
        x = self.dropout2(self.dropout1(self.embedding(x)))
        return x

class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='none', actFunc=nn.ReLU, bn=False, ln=False, name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        bns,lns = [],[]
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
            )
            bns.append(nn.BatchNorm1d(filterSize))
            lns.append(nn.LayerNorm(filterSize))
        if bn:
            self.bns = nn.ModuleList(bns)
        if ln:
            self.lns = nn.ModuleList(lns)
        self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.ln = ln
        self.name = name
    def forward(self, x, mask=None):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [b(i.transpose(1,2)).transpose(1,2) for b,i in zip(self.bns,x)]
        elif self.ln:
            x = [l(i) for l,i in zip(self.lns,x)]
        x = [self.actFunc(i) for i in x]

        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='pool_ensemble':
            maxPool = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
            avgPool = [(i*mask.unsqueeze(dim=-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(dim=-1) for i in x]
            return torch.cat(maxPool+avgPool, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='none':
            return x # => scaleNum * (batchSize × seqLen × filterSize)

class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, ln=False, reduction='none', name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        if ln:
            self.layerNorm =nn.LayerNorm(hiddenSize*2)
        self.ln = ln
        self.reduction = reduction

    def forward(self, x, mask=None):
        # x: batchSizeh × seqLen × feaSize
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if self.ln:
            output = self.layerNorm(output)
        if self.reduction=='pool':
            return torch.max(output, dim=1)[0]
        elif self.reduction=='pool_ensemble':
            maxPool = torch.max(output, dim=1)[0]
            avgPool = (output*mask.unsqueeze(dim=-1)).sum(dim=1) / mask.sum(dim=1).unsqueeze(dim=-1)
            return torch.cat([maxPool,avgPool], dim=1)
        elif self.reduction=='none':
            return output # output: batchSize × seqLen × hiddenSize*2

class SelfAttention_PreLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention_PreLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.WQ = nn.ModuleList([nn.Linear(feaSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(feaSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(feaSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.name = name
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        x = self.layerNorm1(x)
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys    = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values  = [self.WV[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        scores  = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        
        z = [torch.bmm(F.softmax(scores[i], dim=2), values[i]) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batchSize × seqLen × feaSize
        return z

class FFN_PreLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='FFN'):
        super(FFN_PreLN, self).__init__()
        
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        nn.ReLU(),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(z) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(self.layerNorm2(z)) # => batchSize × seqLen × feaSize
        return z+self.dropout(ffnx) # => batchSize × seqLen × feaSize
    
class Transformer_PreLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PreLN, self).__init__()
        self.selfAttn = SelfAttention_PreLN(feaSize, dk, multiNum)
        self.ffn = FFN_PreLN(feaSize, dropout)

    def forward(self, input):
        x, xlen = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z = self.selfAttn(x, xlen) # => batchSize × seqLen × feaSize
        return (self.ffn(x, z),xlen) # => batchSize × seqLen × feaSize

class TransformerLayers_PreLN(nn.Module):
    def __init__(self, seqMaxLen, layersNum, feaSize, dk, multiNum, maxItems=10, dropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_PreLN, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/feaSize)) if i%2==0 else np.cos(pos/10000**(2*i/feaSize)) for i in range(feaSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=True) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PreLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        self.usePos = usePos
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        if self.usePos:
            x = x+self.posEmb
        x = self.dropout(x) # => batchSize × seqLen × feaSize
        return self.transformerLayers((x, xlen)) # => batchSize × seqLen × feaSize

class InvariantPointAttention(nn.Module):
    def __init__(self, feaSize, dk, multiNum, Nqp, Npv, dropout=0.1, name='IPA'):
        super(InvariantPointAttention, self).__init__()
        self.WQh = nn.Linear(feaSize, multiNum*dk, bias=False)
        self.WKh = nn.Linear(feaSize, multiNum*dk, bias=False)
        self.WVh = nn.Linear(feaSize, multiNum*dk, bias=False)

        self.WQhp = nn.Linear(feaSize, multiNum*Nqp*3, bias=False)
        self.WKhp = nn.Linear(feaSize, multiNum*Nqp*3, bias=False)
        self.WVhp = nn.Linear(feaSize, multiNum*Npv*3, bias=False)

        self.Wb = nn.Linear(feaSize, multiNum, bias=False)

        self.WO = nn.Linear(multiNum*feaSize+multiNum*dk+multiNum*Npv*3+multiNum*Npv, feaSize)

        self.wC = np.sqrt(2/(9*Nqp))
        self.wL = np.sqrt(1/3)
        self.gamah = nn.Parameter(torch.zeros(multiNum), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

        self.multiNum = multiNum
        self.dk = dk
        self.Nqp = Nqp
        self.Npv = Npv
    def forward(self, si, zij, Ri, Ti, attn_mask=None):
        # si: B,L,C; zij: B,L,L,C; Ri: B,L,3,3; Ti: B,L,3,1; attn_mask: B,L,L
        B,L,C = si.shape
        qhi = self.WQh(si).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # B,A,L,dk
        khi = self.WKh(si).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # B,A,L,dk
        vhi = self.WVh(si).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # B,A,L,dk
    
        bhij = self.Wb(zij).transpose(2,3).transpose(1,2) # B,A,L,L

        scores1 = qhi@khi.transpose(-1,-2)/np.sqrt(self.dk) + bhij # B,A,L,L

        qhpi = self.WQhp(si).reshape(B,L,self.multiNum,self.Nqp,3,1).transpose(1,2) # B,A,L,Nqp,3,1
        khpi = self.WKhp(si).reshape(B,L,self.multiNum,self.Nqp,3,1).transpose(1,2) # B,A,L,Nqp,3,1
        vhpi = self.WVhp(si).reshape(B,L,self.multiNum,self.Npv,3,1).transpose(1,2) # B,A,L,Npv,3,1

        qhpi_transformed = Ri[:,None,:,None]@qhpi + Ti[:,None,:,None] # B,A,L,Nqp,3,1
        khpi_transformed = Ri[:,None,:,None]@khpi + Ti[:,None,:,None] # B,A,L,Nqp,3,1

        scores2 = F.softplus(self.gamah)[None,:,None,None]*self.wC/2 * torch.sum((qhpi_transformed[:,:,:,None] - khpi_transformed[:,:,None]).squeeze(dim=-1)**2, dim=-1).sum(dim=-1) # B,A,L,L

        scores = self.wL*(scores1 - scores2) # B,A,L,L
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:,None], -2**15+1) # -torch.inf
        alphah = F.softmax(scores, dim=3) # B,A,L,L

        ohi_ = (self.dropout(alphah)[:,:,:,:,None] * zij[:,None]).sum(dim=3) # B,A,L,C 
        ohi  = self.dropout(alphah)@vhi # B,A,L,dk
        
        vhpi_transformed = Ri[:,None,:,None]@vhpi + Ti[:,None,:,None] # B,A,L,Npv,3,1
        ohpi_transformed = (self.dropout(alphah) @ vhpi_transformed.reshape(B,self.multiNum,L,-1)).reshape(B,self.multiNum,L,self.Npv,3,1) # B,A,L,Npv,3,1
        ohpi = Ri.transpose(-1,-2)[:,None,:,None]@(ohpi_transformed-Ti[:,None,:,None]) # B,A,L,Npv,3,1

        si = self.WO(torch.cat([ohi_.transpose(1,2).reshape(B,L,-1),ohi.transpose(1,2).reshape(B,L,-1),ohpi.transpose(1,2).reshape(B,L,-1),ohpi.transpose(1,2).abs().sum(dim=-2).reshape(B,L,-1)],dim=2)) # B,L,C
        return si

class SelfAttention_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, name='selfAttn'):
        super(SelfAttention_PostLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(dropout)
        self.name = name
    def forward(self, qx, kx, vx, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,qL,C = qx.shape
        kvL = kx.shape[1]
        queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        z = self.dropout(alpha) @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,qL,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z

class FFN_PostLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_PostLN, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), # feaSize*4 
                        actFunc(),
                        nn.Dropout(p=dropout),
                        nn.Linear(feaSize*4, feaSize) # feaSize*4
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize

        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × seqLen × feaSize

class Transformer_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PostLN, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum)
        self.ffn = FFN_PreLN(feaSize, dropout)

    def forward(self, input):
        x, xlen = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z = self.selfAttn(x, xlen) # => batchSize × seqLen × feaSize
        return (self.ffn(x, z),xlen) # => batchSize × seqLen × feaSize

class TransformerLayers_PostLN(nn.Module):
    def __init__(self, seqMaxLen, layersNum, feaSize, dk, multiNum, maxItems=10, dropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_PostLN, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/feaSize)) if i%2==0 else np.cos(pos/10000**(2*i/feaSize)) for i in range(feaSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=True) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PostLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        self.usePos = usePos
    def forward(self, x, xlen=None):
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        if self.usePos:
            x = x+self.posEmb
        x = self.dropout(x) # => batchSize × seqLen × feaSize
        return self.transformerLayers((x, xlen)) # => batchSize × seqLen × feaSize

class SelfAttention_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, maxRelativeDist=7, dkEnhance=1, hdnDropout=0.1, name='selfAttn'):
        super(SelfAttention_Realformer, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WK = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=hdnDropout)
        if maxRelativeDist>0:
            self.relativePosEmbK = nn.Embedding(2*maxRelativeDist+1, multiNum)
            self.relativePosEmbB = nn.Embedding(2*maxRelativeDist+1, multiNum)
        self.maxRelativeDist = maxRelativeDist
        self.dkEnhance = dkEnhance
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,L,C = qx.shape

        # print((qx@self.WQ.weight.T + self.WQ.bias).mean())
        # print(self.WQ(qx).mean())

        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        # print(self.WQ.weight.mean(), self.WQ.weight.std(), self.WQ.bias.mean(), self.WQ.bias.std(), qx.mean(), qx.std())
        # print(queries.mean())
        # print(1/0)
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk

        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen
        
        # relative position embedding
        if self.maxRelativeDist>0:
            relativePosTab = torch.abs(torch.arange(0,L).reshape(-1,1) - torch.arange(0,L).reshape(1,-1)).float() # L × L
            relativePosTab[relativePosTab>self.maxRelativeDist] = self.maxRelativeDist+torch.log2(relativePosTab[relativePosTab>self.maxRelativeDist]-self.maxRelativeDist).float()
            relativePosTab = torch.clip(relativePosTab,min=0,max=self.maxRelativeDist*2).long().to(qx.device)
            scores = scores * self.relativePosEmbK(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L) + self.relativePosEmbB(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L)

        # residual attention
        if preScores is not None:
            scores = scores + preScores

        if maskPAD is not None:
            #scores = scores*maskPAD.unsqueeze(dim=1)
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15) # -np.inf

        alpha = self.dropout(F.softmax(scores, dim=3))

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z,scores

class FFN_Realformer(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_Realformer, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        actFunc(),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize

        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × seqLen × feaSize
    
class Transformer_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, maxRelativeDist=7, dropout=0.1, dkEnhance=1, actFunc=nn.GELU):
        super(Transformer_Realformer, self).__init__()
        self.selfAttn = SelfAttention_Realformer(feaSize, dk, multiNum, maxRelativeDist, dkEnhance, dropout)
        self.ffn = FFN_Realformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0) 

class TransformerLayers_Realformer(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, maxRelativeDist=7, embDropout=0.1, hdnDropout=0.1, emb=True, dkEnhance=1, 
                 usePosCNN=False, actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Realformer, self).__init__()

        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Realformer(feaSize, dk, multiNum, maxRelativeDist, hdnDropout, dkEnhance, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=embDropout)
        self.name = name

    def forward(self, x, maskPAD):
        # x: batchSize × seqLen; 
        return self.transformerLayers((x, x, x, None, maskPAD)) # => batchSize × seqLen × feaSize

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class SelfAttention_Pseudoformer(nn.Module):
    def __init__(self, pseudoTknNum, feaSize, dk, multiNum, maxRelativeDist=7, dkEnhance=1, dropout=0.1, name='selfAttn'):
        super(SelfAttention_Pseudoformer, self).__init__()
        self.pseudoTknNum = pseudoTknNum
        self.dk = dk
        self.multiNum = multiNum
        
        self.pseudoAttn = nn.Linear(feaSize, pseudoTknNum)

        self.WQ = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WK = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=dropout)
        if maxRelativeDist>0:
            self.relativePosEmbK = nn.Embedding(2*maxRelativeDist+1, multiNum)
            self.relativePosEmbB = nn.Embedding(2*maxRelativeDist+1, multiNum)
        self.maxRelativeDist = maxRelativeDist
        self.dkEnhance = dkEnhance
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        # obtain the pseudo tokens
        pScore = self.pseudoAttn(vx) # => batchSize × seqLen × pseudoTknNum
        if preScores is not None:
            pScore = pScore + preScores[0]
        if maskPAD is not None:
            pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15)
        pAlpha = self.dropout(F.softmax(pScore, dim=1)) # => batchSize × seqLen × pseudoTknNum

        kx = pAlpha.transpose(1,2) @ kx # => batchSize × pseudoTknNum × feaSize
        vx = pAlpha.transpose(1,2) @ vx # => batchSize × pseudoTknNum × feaSize

        pAlpha = pAlpha.detach()

        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk
        values  = self.WV(vx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × pseudoTknNum

        # relative position embedding
        if self.maxRelativeDist>0:
            relativePosTab = torch.abs(torch.arange(0,L).reshape(-1,1) - torch.arange(0,L).reshape(1,-1)).float() # L × L
            relativePosTab[relativePosTab>self.maxRelativeDist] = self.maxRelativeDist+torch.log2(relativePosTab[relativePosTab>self.maxRelativeDist]-self.maxRelativeDist).float()
            relativePosTab = torch.clip(relativePosTab,min=0,max=self.maxRelativeDist*2).long().to(qx.device) # L × L
            K = self.relativePosEmbK(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L) @ pAlpha.unsqueeze(dim=1) # batchSize × multiNum × seqLen × pseudoTknNum
            b = self.relativePosEmbB(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L) @ pAlpha.unsqueeze(dim=1) # batchSize × multiNum × seqLen × pseudoTknNum
            scores = scores * K + b

        # residual attention
        if preScores is not None:
            scores = scores + preScores[1]

        alpha = self.dropout(F.softmax(scores, dim=3)) # batchSize × multiNum × seqLen × pseudoTknNum

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z,(pScore,scores)

class FFN_Pseudoformer(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_Pseudoformer, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        actFunc(),
                        nn.Dropout(p=dropout),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize

        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × seqLen × feaSize

class Transformer_Pseudoformer(nn.Module):
    def __init__(self, pseudoTknNum, feaSize, dk, multiNum, maxRelativeDist=7, dropout=0.1, dkEnhance=1, actFunc=nn.GELU):
        super(Transformer_Pseudoformer, self).__init__()
        self.selfAttn = SelfAttention_Pseudoformer(pseudoTknNum, feaSize, dk, multiNum, maxRelativeDist, dkEnhance, dropout)
        self.ffn = FFN_Pseudoformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_Pseudoformer(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, pseudoTknNum=64, maxRelativeDist=7, hdnDropout=0.1, dkEnhance=1, 
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Pseudoformer, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Pseudoformer(pseudoTknNum, feaSize, dk, multiNum, maxRelativeDist, hdnDropout, dkEnhance, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD = self.transformerLayers((x, x, x, None, maskPAD))
        return (qx,kx,vx,scores,maskPAD)# => batchSize × seqLen × feaSize

class SelfAttention_Pseudoformer2(nn.Module):
    def __init__(self, pseudoTknNum, feaSize, dk, multiNum, dropout=0.1, name='selfAttn'):
        super(SelfAttention_Pseudoformer2, self).__init__()
        self.pseudoTknNum = pseudoTknNum
        self.dk = dk
        self.multiNum = multiNum
        
        self.pseudoAttn = nn.Linear(feaSize, pseudoTknNum)

        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        # obtain the pseudo tokens
        pScore = self.pseudoAttn(vx) # => batchSize × seqLen × pseudoTknNum
        if preScores is not None:
            pScore = pScore + preScores[0]
        if maskPAD is not None:
            pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15)
        pAlpha = self.dropout(F.softmax(pScore, dim=1)) # => batchSize × seqLen × pseudoTknNum

        qx = pAlpha.transpose(1,2) @ qx # => batchSize × pseudoTknNum × feaSize
        kx = pAlpha.transpose(1,2) @ kx # => batchSize × pseudoTknNum × feaSize
        vx = pAlpha.transpose(1,2) @ vx # => batchSize × pseudoTknNum × feaSize

        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk
        keys    = self.WK(kx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk
        values  = self.WV(vx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × pseudoTknNum × pseudoTknNum

        # residual attention
        if preScores is not None:
            scores = scores + preScores[1]

        alpha = self.dropout(F.softmax(scores, dim=3)) # batchSize × multiNum × pseudoTknNum × pseudoTknNum

        z = alpha @ values # => batchSize × multiNum × pseudoTknNum × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × pseudoTknNum × multiNum*dk

        z = self.WO(z) # => batchSize × pseudoTknNum × feaSize

        pAlpha_ = self.dropout(F.softmax(pScore, dim=2)) # => batchSize × seqLen × pseudoTknNum
        z = pAlpha_ @ z # => batchSize × seqLen × feaSize
        return z,(pScore,scores)

class Transformer_Pseudoformer2(nn.Module):
    def __init__(self, pseudoTknNum, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(Transformer_Pseudoformer2, self).__init__()
        self.selfAttn = SelfAttention_Pseudoformer2(pseudoTknNum, feaSize, dk, multiNum, dropout)
        self.ffn = FFN_Pseudoformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_Pseudoformer2(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, pseudoTknNum=64, hdnDropout=0.1,  
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Pseudoformer2, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Pseudoformer2(pseudoTknNum, feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD = self.transformerLayers((x, x, x, None, maskPAD))
        return (qx,kx,vx,scores,maskPAD)# => batchSize × seqLen × feaSize

class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ELU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                        nn.ELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                    )
        self.name = name
    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)

class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=10, dilaSizeList=[1,2,4,8,16], dropout=0.15, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize,filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i%len(dilaSizeList)],filterSize,dropout=dropout))
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x) # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1,2)).transpose(1,2) # => batchSize × seqLen × filterSize
        return F.elu(x) # => batchSize × seqLen × filterSize

class GCN(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, resnet=False, name='GCN', actFunc=nn.ReLU):
        super(GCN, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenSizeList):
            hiddens.append(nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet
    def forward(self, x, L):
        # x: batchSize × nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for h,bn in zip(self.hiddens,self.bns):
            a = h(torch.matmul(L,x)) # => batchSize × nodeNum × os
            if self.bnEveryLayer:
                if len(L.shape)==3:
                    a = bn(a.transpose(1,2)).transpose(1,2)
                else:
                    a = bn(a)
            a = self.actFunc(a)
            if self.dpEveryLayer:
                a = self.dropout(a)
            if self.resnet and a.shape==x.shape:
                a += x
            x = a
        a = self.out(torch.matmul(L, x)) # => batchSize × nodeNum × outSize
        if self.outBn:
            if len(L.shape)==3:
                a = self.bns[-1](a.transpose(1,2)).transpose(1,2)
            else:
                a = self.bns[-1](a)
        if self.outAct: a = self.actFunc(a)
        if self.outDp: a = self.dropout(a)
        if self.resnet and a.shape==x.shape:
            a += x
        x = a
        return x

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, inBn=False, bnEveryLayer=False, dpEveryLayer=False, outBn=False, outAct=False, outDp=False, inDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        self.sBn = nn.BatchNorm1d(inSize)
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.inBn = inBn
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.inDp = inDp
    def forward(self, x):
        if self.inBn:
            x = self.sBn(x)
        if self.inDp:
            x = self.dropout(x)
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape)==2 else bn(x.transpose(1,2)).transpose(1,2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape)==2 else self.bns[-1](x.transpose(1,2)).transpose(1,2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class SingularAttention(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, linearLevel='l1', name='selfAttn'):
        super(SingularAttention, self).__init__()
        self.r = r
        self.dk = dk
        self.multiNum = multiNum
        
        if linearLevel=='l0':
            self.pseudoAttn = nn.Linear(feaSize, r, bias=False)
        elif linearLevel=='l1':
            self.pseudoAttn = nn.Linear(feaSize, r, bias=False)
        elif linearLevel=='l2':
            self.fQ = nn.Linear(feaSize, r, bias=False)
            self.fKV = nn.Linear(feaSize, r, bias=False)
        elif linearLevel=='l3':
            self.fKV = nn.Linear(feaSize, r, bias=False)
        
        if linearLevel=='l0':
            self.WQ = nn.Linear(feaSize, self.dk*multiNum, bias=False)
            self.WK = nn.Linear(feaSize, self.dk, bias=False)
            self.WV = nn.Linear(feaSize, self.dk, bias=False)
            self.WO = nn.Linear(self.dk*multiNum, feaSize, bias=False)
        else:
            self.WQ = nn.Linear(feaSize, self.dk*multiNum, bias=False)
            self.WK = nn.Linear(feaSize, self.dk*multiNum, bias=False)
            self.WV = nn.Linear(feaSize, self.dk*multiNum, bias=False)
            self.WO = nn.Linear(self.dk*multiNum, feaSize, bias=False)
        self.linearLevel = linearLevel
        self.dropout = nn.Dropout(p=dropout)
        self.name = name

        # self.I = nn.Parameter(torch.eye(self.r, dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.ones_minus_I = nn.Parameter((torch.ones((self.r,self.r), dtype=torch.bool) ^ torch.eye(self.r, dtype=torch.bool)).unsqueeze(0).unsqueeze(0), requires_grad=False)
        
    def forward(self, qx, kx, vx, addLoss, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        # obtain the pseudo tokens
        B,qL,C = qx.shape

        if self.linearLevel=='l0':
            pScore = self.pseudoAttn(vx) # => batchSize × seqLen × r

            pAlpha = F.tanh(pScore).transpose(-1,-2) # => batchSize × r × seqLen
            
            # kx = self.dropout(pAlpha)[:,:,:,None] * kx[:,None,:,:] # => batchSize × r × seqLen × feaSize
            vx = self.dropout(pAlpha)[:,:,:,None] * vx[:,None,:,:] # => batchSize × r × seqLen × feaSize

            # kx,_ = torch.cumsum(kx, dim=2).max(dim=1)
            vx,_ = torch.cumsum(vx, dim=2).max(dim=1)

            queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
            keys    = self.WK(vx).reshape(B,1,qL,self.dk) # => batchSize × 1 × seqLen × dk
            values  = self.WV(vx).reshape(B,1,qL,self.dk) # => batchSize × 1 × seqLen × dk
        
            diff = 1
            scores = (queries[:,:,diff:,:]*keys[:,:,:-diff,:]).sum(dim=-1, keepdims=True) / np.sqrt(self.dk) # => batchSize × multiNum × (seqLen-diff) × 1
            alpha = F.sigmoid(scores)

            z = alpha * values[:,:,:-diff,:] # => batchSize × multiNum × (seqLen-diff) × dk

            z = z.transpose(1,2).reshape(B,qL-diff,-1) # => batchSize × (seqLen-diff) × multiNum*dk

            z = self.WO(z) # => batchSize × r × (seqLen-diff) × feaSize
            z = F.pad(z, (0,0,diff,0,0,0), mode='constant', value=0)

        elif self.linearLevel=='l1':
            pScore = self.pseudoAttn(vx) # => batchSize × seqLen × r
            if maskPAD is not None:
                pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)

            pAlpha = F.softmax(pScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            pAlpha_ = F.softmax(pScore, dim=-1) # => batchSize × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean((((pAlpha @ pAlpha.transpose(-1,-2))*self.ones_minus_I[0])**2))

            qx = self.dropout(pAlpha) @ qx # => batchSize × r × feaSize
            kx = self.dropout(pAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(pAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean(alpha * self.ones_minus_I)
                # addLoss += torch.mean((alpha * self.ones_minus_I).sum(-1))

            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            z = z.transpose(1,2).reshape(B,self.r,-1) # => batchSize × r × multiNum*dk

            z = self.WO(z) # => batchSize × r × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean((((pAlpha_.transpose(-1,-2) @ pAlpha_)*self.ones_minus_I[0])**2))

            z = self.dropout(pAlpha_) @ z # => batchSize × seqLen × feaSize

        elif self.linearLevel=='l2':
            qScore,kvScore = self.fQ(qx),self.fKV(vx) # => batchSize × qL × r, batchSize × kvL × r
            if maskPAD is not None:
                qScore = qScore.masked_fill((maskPAD[:,:,0]==0).unsqueeze(dim=-1), -2**15+1)
                kvScore = kvScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                
            qAlpha,kvAlpha = F.softmax(qScore, dim=-2).transpose(-1,-2),F.softmax(kvScore, dim=-2).transpose(-1,-2) # => batchSize × r × qL, batchSize × r × kvL
            qAlpha_,kvAlpha_ = F.softmax(qScore, dim=-1),F.softmax(kvScore, dim=-1) # => batchSize × qL × r,batchSize × kvL × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((kvAlpha@kvAlpha.transpose(-1,-2))*self.ones_minus_I[0])**2)

            qx = self.dropout(qAlpha) @ qx # => batchSize × r × feaSize
            kx = self.dropout(kvAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(kvAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean((alpha * self.ones_minus_I))
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            z = z.transpose(1,2).reshape(B,self.r,-1) # => batchSize × r × multiNum*dk

            z = self.WO(z) # => batchSize × r × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += (torch.mean(((qAlpha_.transpose(-1,-2) @ qAlpha_)*self.ones_minus_I[0])**2) + torch.mean(((kvAlpha_.transpose(-1,-2) @ kvAlpha_)*self.ones_minus_I[0])**2)) / 2

            z = self.dropout(qAlpha_) @ z # => batchSize × seqLen × feaSize

        elif self.linearLevel=='l3':
            kvScore = self.fKV(vx) # => batchSize × qL × r, batchSize × kvL × r
            if maskPAD is not None:
                kvScore = kvScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                
            kvAlpha = F.softmax(kvScore, dim=-2).transpose(-1,-2) # => batchSize × r × kvL
            kvAlpha_ = F.softmax(kvScore, dim=-1) # => batchSize × kvL × r

            qx = qx # => batchSize × qL × feaSize
            kx = self.dropout(kvAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(kvAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × qL × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × qL × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × qL × r
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × qL × dk
            z = z.transpose(1,2).reshape(B,qL,-1) # => batchSize × qL × multiNum*dk

            z = self.WO(z) # => batchSize × qL × feaSize

        return z,scores,addLoss

class SingularformerEncoderBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU, linearLevel='l1'):
        super(SingularformerEncoderBlock, self).__init__()
        self.selfAttn = SingularAttention(r, feaSize, dk, multiNum, dropout, linearLevel=linearLevel)
        # self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        # self.ffn = FFN_Pseudoformer(feaSize, dropout, actFunc)
        self.ffn = FFN_GatedLinearUnit(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD,addLoss = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        zq,preScores,addLoss = self.selfAttn(qx,kx,vx,addLoss,preScores,maskPAD) # => batchSize × seqLen × feaSize
        # zq = self.selfAttn(qx,kx,vx,maskPAD) # => batchSize × seqLen × feaSize

        x = self.ffn(qx, zq)
        return (x, x, x, preScores,maskPAD,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)
class SingularformerEncoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1,  
                 actFunc=nn.GELU, linearLevel='l1', name='textTransformer'):
        super(SingularformerEncoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerEncoderBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc, linearLevel)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD,addLoss = self.transformerLayers((x, x, x, None, maskPAD,0))
        return (qx,kx,vx,scores,maskPAD,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class SingularformerDecoderBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU, linearLevel='l1'):
        super(SingularformerDecoderBlock, self).__init__()
        # self.selfAttn1 = SingularAttention(multiNum, feaSize, dk, multiNum, dropout, linearLevel='l0')
        # self.selfAttn1 = SelfAttention_Linear(feaSize, dk, multiNum, dropout)
        self.selfAttn1 = SelfAttention_CumNext(feaSize, dk, multiNum)
        # self.selfAttn1 = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.selfAttn2 = SingularAttention(r, feaSize, dk, multiNum, dropout, linearLevel='l3')
        # self.selfAttn2 = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.layernorm = nn.LayerNorm([feaSize])
        # self.ffn = FFN_Pseudoformer(feaSize, dropout, actFunc)
        self.ffn = FFN_GatedLinearUnit(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input, predict=False):
        qx,kx,vx,preScores,maskPAD1,maskPAD2,addLoss = input
        preScores1,preScores2 = preScores
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        # qx_,_,addLoss = self.selfAttn1(qx,qx,qx,addLoss,0,maskPAD1) # => batchSize × seqLen × feaSize
        # qx = self.layernorm(qx+qx_)
        z,preScores1 = self.selfAttn1(qx,qx,qx,preScores1)
        qx = self.layernorm(qx+z) # => batchSize × seqLen × feaSize
        # qx = self.layernorm(qx+self.selfAttn1(qx,qx,qx,maskPAD1)) # => batchSize × seqLen × feaSize
        
        zq,preScores2,addLoss = self.selfAttn2(qx,kx,vx,addLoss,preScores2,maskPAD2) # => batchSize × seqLen × feaSize
        # print(zq.shape,zq[0,[0]])
        # zq = self.selfAttn2(zq,kx,vx,maskPAD2)

        x = self.ffn(qx, zq)
        return (x, kx, vx, (preScores1,preScores2),maskPAD1,maskPAD2,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class SingularformerDecoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1, diffs=None, 
                 actFunc=nn.GELU, linearLevel='l1', name='textTransformer'):
        super(SingularformerDecoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerDecoderBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc, linearLevel)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, xRef, maskPAD1, maskPAD2):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss = self.transformerLayers((x, xRef, xRef, (0,0), maskPAD1,maskPAD2,0))
        return (qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class TransformerEncoderBlock(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(TransformerEncoderBlock, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.ffn = FFN_PostLN(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD,addLoss = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        zq = self.selfAttn(qx,kx,vx,maskPAD) # => batchSize × seqLen × feaSize

        x = self.ffn(qx, zq)
        return (x, x, x, preScores,maskPAD,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerEncoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, hdnDropout=0.1,  
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerEncoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, TransformerEncoderBlock(feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD,addLoss = self.transformerLayers((x, x, x, None, maskPAD,0))
        return (qx,kx,vx,scores,maskPAD,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class TransformerDecoderBlock(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(TransformerDecoderBlock, self).__init__()
        self.selfAttn1 = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.selfAttn2 = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.layernorm = nn.LayerNorm([feaSize])
        self.ffn = FFN_PostLN(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input, predict=False):
        qx,kx,vx,preScores,maskPAD1,maskPAD2,addLoss = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        qx = self.layernorm(qx+self.selfAttn1(qx,qx,qx,maskPAD1)) # => batchSize × seqLen × feaSize
        
        zq = self.selfAttn2(qx,kx,vx,maskPAD2)

        x = self.ffn(qx, zq)
        return (x, kx, vx, preScores,maskPAD1,maskPAD2,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerDecoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, hdnDropout=0.1,  
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerDecoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, TransformerDecoderBlock(feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, xRef, maskPAD1, maskPAD2):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss = self.transformerLayers((x, xRef, xRef, None, maskPAD1,maskPAD2,0))
        return (qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class SelfAttention_CumNext(nn.Module):
    def __init__(self, feaSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention_CumNext, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum, bias=False)
        self.WK = nn.Linear(feaSize, self.dk*multiNum, bias=False)
        self.WV = nn.Linear(feaSize, self.dk*multiNum, bias=False)
        self.WO = nn.Linear(self.dk*multiNum, feaSize, bias=False)

        self.div = torch.nn.Parameter(torch.arange(1,4097)[None,:,None].float(), requires_grad=False)
        
        self.name = name
    def forward(self, qx, kx, vx, preScores=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,qL,C = qx.shape
        kvL = kx.shape[1]

        vx = torch.cumsum(vx, dim=1) / self.div[:,:kvL]
        # vx = (vx-vx.mean(dim=-1, keepdims=True)) / (vx.std(dim=-1, keepdims=True) + 1e-6)

        kx = vx

        queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk

        diff = 1
        if qL>diff:
            scores = (queries[:,:,diff:,:]*keys[:,:,:-diff,:]).sum(dim=-1, keepdims=True) / np.sqrt(self.dk) # => batchSize × multiNum × (seqLen-1)
            
            if preScores is not None:
                scores = scores+preScores

            alpha = F.sigmoid(scores)
            z = alpha * values[:,:,:-diff,:] # => batchSize × multiNum × (seqLen-1) × dk

            z = z.transpose(1,2).reshape(B,qL-diff,-1) # => batchSize × seqLen × multiNum*dk

            z = self.WO(z) # => batchSize × (seqLen-1) × feaSize
            return F.pad(z, (0,0,diff,0,0,0), mode='constant', value=0),scores # => batchSize × seqLen × feaSize
        else:
            return torch.zeros_like(qx),0 # => batchSize × seqLen × feaSize

class SingularformerGLUEncoderBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.SiLU):
        super(SingularformerGLUEncoderBlock, self).__init__()
        self.hdnSize = feaSize*2
        self.pseudoAttn = nn.Linear(feaSize, r)
        self.WQ = nn.Sequential(nn.Linear(feaSize, dk), actFunc())
        self.WK = nn.Sequential(nn.Linear(feaSize, dk), actFunc())
        self.WV = nn.Sequential(nn.Linear(feaSize, self.hdnSize), actFunc())

        # self.offsetscale = OffsetScale(dk, heads=2)
        self.layerNorm = nn.LayerNorm([feaSize])

        self.WU = nn.Sequential(nn.Linear(feaSize, self.hdnSize),actFunc()) # 
        self.WO = nn.Sequential(nn.Linear(self.hdnSize, feaSize))

        self.dropout = nn.Dropout(dropout)

        self.r = r
        self.dk = dk

        self._reset_parameters()
    def forward(self, input):
        x,_,_,preScores,maskPAD,addLoss = input

        B,L,C = x.shape

        gate = self.WU(x) # => batchSize × seqLen × hdnSize

        pScore = self.pseudoAttn(x) # => batchSize × seqLen × r
        if maskPAD is not None:
            pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)

        pAlpha = F.softmax(pScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
        pAlpha_ = F.softmax(pScore, dim=-1) # => batchSize × seqLen × r

        px = self.dropout(pAlpha) @ x # => batchSize × r × feaSize

        queries = self.WQ(px).reshape(B,self.r,self.dk) # => batchSize × r × dk
        keys    = self.WK(px).reshape(B,self.r,self.dk) # => batchSize × r × dk
        values  = self.WV(px).reshape(B,self.r,self.hdnSize) # => batchSize × r × hdnSize

        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × r × r self.r*
        # residual attention
        if preScores is not None:
            scores = scores + preScores
            preScores = scores

        alpha = F.softmax(scores, dim=2) # F.relu(scores)**2 # batchSize × r × r
        z = self.dropout(alpha) @ values # => batchSize × r × hdnSize
        z = self.dropout(pAlpha_) @ z # => batchSize × seqLen × hdnSize

        z = gate * z
        z = self.WO(z) # => batchSize × seqLen × feaSize

        x = self.layerNorm(x+self.dropout(z))
        return (x, x, x, preScores,maskPAD,addLoss) # => batchSize × seqLen × feaSize
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class SingularformerGLUDecoderBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.SiLU):
        super(SingularformerGLUDecoderBlock, self).__init__()
        
        self.qSelfAttn = SelfAttention_CumNext(feaSize, dk, multiNum)
        self.qLayerNorm = nn.LayerNorm(feaSize)

        self.hdnSize = feaSize*2
        self.pseudoAttn = nn.Linear(feaSize, r)
        self.WQ = nn.Sequential(nn.Linear(feaSize, dk),actFunc())
        self.WK = nn.Sequential(nn.Linear(feaSize, dk),actFunc())
        self.WV = nn.Sequential(nn.Linear(feaSize, self.hdnSize),actFunc())

        # self.offsetscale = OffsetScale(dk, heads=2)
        self.layerNorm = nn.LayerNorm([feaSize])

        self.WU = nn.Sequential(nn.Linear(feaSize, self.hdnSize),actFunc())
        self.WO = nn.Sequential(nn.Linear(self.hdnSize, feaSize),nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

        self.r = r
        self.dk = dk

        self._reset_parameters()
    def forward(self, input):
        qx,kx,vx,preScores,maskPAD1,maskPAD2,addLoss = input
        B,qL,C = qx.shape
        kvL = kx.shape[1]

        preScores1,preScores2 = preScores

        z,preScores1 = self.qSelfAttn(qx,qx,qx,preScores=preScores1)
        qx = self.qLayerNorm(qx+z) # => batchSize × seqLen × feaSize

        gate = self.WU(qx) # => batchSize × kvL × hdnSize

        pScore = self.pseudoAttn(vx) # => batchSize × kvL × r
        if maskPAD2 is not None:
            pScore = pScore.masked_fill((maskPAD2[:,0]==0).unsqueeze(dim=-1), -2**15+1)

        pAlpha = F.softmax(pScore, dim=-2).transpose(-1,-2) # => batchSize × r × kvL
        
        pkvx = self.dropout(pAlpha) @ vx # => batchSize × r × feaSize

        queries = self.WQ(qx).reshape(B,qL,self.dk) # => batchSize × qL × dk
        keys    = self.WK(pkvx).reshape(B,self.r,self.dk) # => batchSize × r × dk
        values  = self.WV(pkvx).reshape(B,self.r,self.hdnSize) # => batchSize × r × hdnSize

        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × qL × r self.r*
        # residual attention
        if preScores2 is not None:
            scores = scores + preScores2
            preScores2 = scores

        alpha = F.softmax(scores, dim=2) # F.relu(scores)**2 # batchSize × qL × r
        z = self.dropout(alpha) @ values # => batchSize × qL × hdnSize

        z = gate * z
        z = self.WO(z) # => batchSize × seqLen × feaSize

        qx = self.layerNorm(qx+self.dropout(z))
        return (qx, kx, vx, (preScores1,preScores2),maskPAD1,maskPAD2,addLoss) # => batchSize × seqLen × feaSize
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class SingularformerGLUEncoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1,  
                 actFunc=nn.SiLU, name='encoder'):
        super(SingularformerGLUEncoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerGLUEncoderBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD,addLoss = self.transformerLayers((x, x, x, 0, maskPAD,0))
        return (qx,kx,vx,scores,maskPAD,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class SingularformerGLUDecoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1,  
                 actFunc=nn.SiLU, name='decoder'):
        super(SingularformerGLUDecoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerGLUDecoderBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, xRef, maskPAD1, maskPAD2):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss = self.transformerLayers((x, xRef, xRef, (0,0), maskPAD1,maskPAD2,0))
        return (qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class PseudoLabelAttention(nn.Module):
    def __init__(self, hdnSize, pseLabNum=64, dropout=0.1, name='pseLabAttn'):
        super(PseudoLabelAttention, self).__init__()
        self.LN1 = nn.LayerNorm([hdnSize])
        self.pseAttnLayer = nn.Sequential(
                               nn.Linear(hdnSize, hdnSize*4),
                               nn.ReLU(),
                               nn.Linear(hdnSize*4, pseLabNum)
                            )
        self.LN2 = nn.LayerNorm([hdnSize])
        self.ffn = nn.Sequential(
                        nn.Linear(hdnSize, hdnSize*4),
                        nn.ReLU(),
                        nn.Linear(hdnSize*4, hdnSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, mask=None):
        # x: B,L,C; mask: B,L
        scores = self.pseAttnLayer(self.LN1(x)) # B,L,k

        if mask is not None:
            scores = scores.masked_fill(~mask[:,:,None], -2**15+1) # -torch.inf

        pseAttn = self.dropout(F.softmax(scores, dim=1)) # B,L,k

        x = pseAttn.transpose(-1,-2)@x # B,k,C
        ffnx = self.ffn(self.LN2(x))
        return x + self.dropout(ffnx)

class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm([feaSize])
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.layerNorm(x))

class ValidBCELoss(nn.Module):
    def __init__(self, threshold=-0.5):
        super(ValidBCELoss, self).__init__()
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, y_logit, y):
        y = y.reshape(-1).float()
        y_logit = y_logit.reshape(-1)
        isValid = y>self.threshold
        return self.criterion(y_logit[isValid], y[isValid])

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, ignoreIdx=-100, gama=2, smoothing=0.1, classes=None, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        if classes is not None:
            smoothing *= classes/(classes-1)
        self.ignoreIdx = ignoreIdx
        self.gama = gama
        self.smoothing = smoothing
        self.logit = logit
    def forward(self, Y_pre, Y):
        isValid = Y!=self.ignoreIdx
        Y_pre,Y = Y_pre[isValid],Y[isValid]
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        
        CE = torch.log(P)
        SL = torch.log(Y_pre).mean(dim=-1)

        return -((1-P)**self.gama * ((1-self.smoothing)*CE + self.smoothing*SL)).mean()

# class SmoothCrossEntropyLoss(nn.Module):
#     def __init__(self, smoothTab, ignore_index=-999, name='smoothCrossEntropyLoss'):
#         super(SmoothCrossEntropyLoss, self).__init__()
#         self.ignore_index = ignore_index
#         self.smoothTab = smoothTab # C × C
#         self.name = name
#     def forward(self, pred, target):
#         # pred: B × C; target: B
#         isValid = target!=self.ignore_index
#         pred,target = pred[isValid],target[isValid]
#         logprobs = F.log_softmax(pred, dim=1)
#         smoothTarget = self.smoothTab[target] # => B × C
#         loss = -torch.sum(logprobs*smoothTarget, dim=1)
#         return loss.mean()

class SmoothCrossEntropyLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, ignoreIdxs=[]):
        """
        Constructor for the SmoothCrossEntropyLoss module.
        :param smoothing: label smoothing factor
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.ignoreIdxs = ignoreIdxs

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs[:,(~torch.isinf(logprobs)).all(dim=0)].mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, gama=0.2, name='contrastiveLoss'):
        super(ContrastiveLoss, self).__init__()
        self.gama = gama
        self.name = name
    def forward(self, vectors, labels):
        # vectors: B × H; labels: B;
        dist = vectors@vectors.T # => B × B
        dist = torch.exp(dist/self.gama)
        loss = 0

        for lab in set(labels):
            posIdx = labels==lab
            tmp = dist[posIdx]
            pos = tmp[:,posIdx][~torch.eye(len(tmp), dtype=torch.bool)].reshape(len(tmp),-1) # pn × pn
            neg = torch.sum(tmp[:,~posIdx], dim=1, keepdims=True) # pn × 1
            loss += -torch.sum(torch.mean(torch.log2(pos / (pos+neg)), dim=1))

        return loss / len(labels)

class ProteinDrugInteractionLoss(nn.Module):
    def __init__(self, ignore_index):
        super(ProteinDrugInteractionLoss, self).__init__()
        self.mlmProteinLoss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mlmDrugLoss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dtaLoss = nn.MSELoss()
    def forward(self, y_logit, y_true):
        proteinLogit, drugLogit, dtaLogit = y_logit['proteinLogit'],y_logit['drugLogit'],y_logit['dtaLogit']
        proteinTrue, drugTrue, dtaTrue = y_true['pSeqArr'],y_true['dSeqArr'],y_true['affinities']
        _,_,amNum = proteinLogit.shape
        proteinLogit = proteinLogit.reshape(-1,amNum)
        proteinTrue = proteinTrue.reshape(-1)
        _,_,atNum = drugLogit.shape
        drugLogit = drugLogit.reshape(-1,atNum)
        drugTrue = drugTrue.reshape(-1)

        return self.mlmProteinLoss(proteinLogit, proteinTrue) + self.mlmDrugLoss(drugLogit, drugTrue) + self.dtaLoss(F.sigmoid(dtaLogit), F.sigmoid(dtaTrue))

