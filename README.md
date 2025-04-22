

# Customeize read data path  :
     - file dataset.__init__
     - line 106
     - function make_dataset

# Customize how input read and transforms to fit mode input:
     -- file dataset.utils
     -- line 242
     -- function __next__

# Dataset format
ANPM , N is Class, M is sample id , A denoted for Action (class) P is person customized for our case 
```
--root
     --train
          -rgb
               - __A1P2.avi
               - __A2P2.avi
               - __A3P5.avi
               - __A__P__.avi
     --test
     --val
```
# Training
```bash
CUDA_VISIBLE_DEVICES=1,2,6 python /work/21013187/SAM-SLR-v2/phuoc_src/train.py
``` 
# Requirements 
- Tranformers >= 4.5 (in my case)
- Python 3.11 (my case)
