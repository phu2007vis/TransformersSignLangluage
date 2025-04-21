

# Thay đổi cách đọc đường dẫn :
     - file dataset.__init__
     - line 10606
     - function make_dataset

# Thay đổi cách xử lý dữ liệu từ đường dẫndẫn  ở:
     -- file dataset.utils
     -- line 242
     -- function __next__

# Dataset format
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