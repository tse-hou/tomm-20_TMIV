# DRL algorithm in tomm'20_TMIV

- Train the model:
```
usage: DRL_wrapper.py [-h] video_type is_training gpu_idx model_name

DRL warper

positional arguments:
  video_type   ERP or PTP
  is_training  is training (1) or testing (0)
  gpu_idx      which gpu want to use
  model_name   the name of model

optional arguments:
  -h, --help   show this help message and exit
```

- utils.py need to be modified to find right video sequence for ERP/PTP model training
  - mark by "# train and testing setting:"
  - done by wrapper
