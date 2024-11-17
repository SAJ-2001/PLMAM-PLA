# PLMAM-PLA: a novel deep learning model based on pretrained language model and attention mechanism for protein-ligand binding affinity prediction
## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---
## 1. Introduction

We
developed a novel sequence-based deep learning method, called PLMAM-PLA, to predict protein-ligand binding affinity.
PLMAM-PLA is constructed by integrating local and global sequence features. The PLMAM-PLA model consists of a
feature extraction module,a feature enhancement module, a feature enhancement module, and an output module. The
feature extraction module uses a CNN sub-module to obtain the initial local features of the sequence and a pretrained
language model to obtain the global features. The feature enhancement module extracts higher-order local and global
features. The feature fusion module learns protein-ligand interactions and integrates all the features. The proposed
model is trained and tested using the PDBbind v2016 dataset. We compared PLMAM-PLA with the latest state-of-theart methods and analyzed the effectiveness of different parts of the model. The results show that the proposed model
outperforms other deep learning models.


## 2. Python Environment

Python 3.9 and packages version:

- pytorch==2.2.1
- tqdm==4.66.2                            
- torchvision==0.17.1    
- transformers==4.22.2
- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.4.1
- scipy==1.12.0 

## 3. Project Structure

### 3.1 **Dataset**

   For this study, we choose PDBbind database includes a set of experimentally validated protein-ligand binding affinities from the Protein Data Bank, expressed as -logKi, -logKd, or -logIC50. Three datasets (the general set, the refined set, and the core 2016 set) from PDBbind version 2016 were used in our work. The general set contains 9226 collected protein-ligand complexes. The refined set contains a total of 4057 high-quality affinity data and complexes. The core 2016 set is typically used as a high-quality benchmark and includes a diverse set of structural and binding data for evaluating various docking methods. To ensure that there was no data overlap between the three datasets, 290 protein-ligand complexes from the core 2016 set were removed from the refined set. The final general set contains 9221 complexes, the refined set contains 3685 complexes, and the core 2016 set contains 290 complexes. Then, we randomly selected 1000 complexes in the refined set as the validation set, and merged the remaining complexes into the universal set as the training set. All in all, we have 11906 training samples, 1000 validation samples, and 290 test samples.

### 3.2 **Model**
   -  The overall architectures of PLMAM-PLA is presented in the following figure, which consists of a Feature extraction module, a Feature enhancement module, a Feature fusion module and a Output module.
   -  ![Model Architecture](https://github.com/SAJ-2001/PLMAM-PLA/blob/main/PLMAM-PLA.jpg)
   -  best_model.pt is the PLMAM-PLA model that is trained on the training subest of the PDBbind dataset.
   -  The ESM-2 model is available at (https://github.com/facebookresearch/esm) and Molformer model is available at (https://huggingface.co/ibm/MoLFormer-XL-both-10pct).
   -   To load the model from Huggingface, we can use the following code:
   -   ```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
outputs.pooler_output
     ```shell

   

