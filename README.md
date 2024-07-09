# NeurIPS2024-BELKA-ensemble-of-fingerprint-atom-SMILES-features
6 models that use the fingerprint/atom/SMILES-level features for representating molecules.

## USAGE

```python
python train_BELKA.py --modelType FprMLP --EMA True            # for 7 fingerprint-based MLP model
python train_BELKA.py --modelType DeepFM/DeepFM2 --EMA True    # for 4 fingerprint-based DeepFM model
python train_BELKA.py --modelType PseLabAttn --EMA True        # for SMILES/ECFP/atom features-based RNN-Transformer model
python train_BELKA.py --modelType GraphMLP/GraphMLP2 -EMA True # for SMILES/FCFP/atom features-based GNN model
python train_BELKA_lgb.py                                      # for 7 fingerprint-based lgb model
```

FprMLP/DeepFM/DeepFM2 are all based on the molecular fingerprint features only, and achieve **0.620~0.645** in validation(15-fold) and **0.432** in public LB;

PseLabAttn/GraphMLP are feature-mixture model, achieving **0.650** in validation(15-fold) and **0.458/0.398** in public LB; (this GNN didn't consider the bond type)

lgb is fingerprint-based lightgbm model, achieving **0.615** in validation(15-fold) and **0.377** in public LB;

ensemble all of them can lead to about **0.480** in public LB. 
