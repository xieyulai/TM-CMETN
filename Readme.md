# TS-CMETN
Two-Stage Cross-Modal Encoding Transformer Network (TS-CMETN) for Dense Video Caption.

Code and data for the paper [Tri-modal Dense Video Captioning Based on Fine-Grained Aligned Text and Anchor-Free Event Proposals Generator]()

Please cite the following paper if you use this repository in your research.
```
@article{doi:10.1142/S021800142255014X,
author = {Niu, Jingjing and Xie, Yulai and Zhang, Yang and Zhang, Jinyu and Zhang, Yanfei and Lei, Xiao and Ren, Fang},
title = {Tri-modal Dense Video Captioning Based on Fine-Grained Aligned Text and Anchor-Free Event Proposals Generator},
journal = {International Journal of Pattern Recognition and Artificial Intelligence},
volume = {0},
number = {ja},
pages = {null},
year = {0},
doi = {10.1142/S021800142255014X},

URL = { 
        https://doi.org/10.1142/S021800142255014X
    
},
eprint = { 
        https://doi.org/10.1142/S021800142255014X
    
}

```


## Preparation
Download the premise files from [BAIDU PAN](https://pan.baidu.com/s/1SIyud2YkImbjswnEkZ1gog) (pw:`zfdf`) and put them in the root directory.

Keep your root dir like this:
```
___ checkpoint
___ data -> ../CAPTION_DATA
___ datasets
___ epoch_loops
___ evaluation
___ log
___ loss
___ main_fcos.py
___ model
___ Readme.md
___ run_caption.sh
___ run_eval_props.sh
___ run_proposal.sh
___ scripts
___ submodules 
___ tmp
___ utilities
___ .vector_cache

```

## Caption Training
```
sh run_caption.sh

```
## Proposal Training
```
sh run_proposal.sh

```

## Evaluation
```
sh run_eval_props.sh	
```
