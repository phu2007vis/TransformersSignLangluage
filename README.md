

# TODO
- Upload pretrained weights
- Fuse depth info
- Upload some samples of the dataset
- If you found this repository helpful, please consider starring.

# Default Dataset format
ANPM, N is Class, M is sample id, A denotes Action (class), P is person customized for our case 
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
# Manually pre-process the cache RGB for boost training read data 
- If not run, the code will auto generate the cache at the first epoch, but slower atthe  first epoch
```bash
python helper_fn/generate_cache.py --root_folder=your/root/folder
```
# Customize read data path  :
     - file dataset.__init__
     - function make_dataset

# Customize how input is read and transformed to fit the mode input:
     -- file dataset.utils   function __next__
     -- file dataset.utils   function collate_fn
# Training
```bash
CUDA_VISIBLE_DEVICES=1,2,6 python /work/21013187/SAM-SLR-v2/phuoc_src/train.py
``` 
# Change config batch size,data root
- config/landmarks.yaml
# Requirements 
- Tranformers >= 4.5 (in my case)
- Python 3.11 (my case)
