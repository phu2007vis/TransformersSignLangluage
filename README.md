

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
               - __A1P2.avi (Class 1)
               - __A1P2.avi (class 1)
               - __A2P2.avi (class 2)
               - __A3P5.avi (class 3)
               - __A__P__.avi
          -npy (landmark)
               - __A1P2.npy (T,N,C) c = 3 (x,y,z) N is number of keypoints
               ...
     --test
     --val
```
# Manually preproces cache rgb for boost trainning read data 
- if not run the code will auto generate the cache at the fisrt epoch but slower at first epoch
```bash
python helper_fn/generate_cache.py --root_folder=your/root/folder
```
# Training
```bash
CUDA_VISIBLE_DEVICES=1,2,6 python /work/21013187/SAM-SLR-v2/phuoc_src/train.py
``` 
# Requirements 
- Tranformers >= 4.5 (in my case)
- Python 3.11 (my case)
