# nerf_project
NeRF code for colleagues

## Requirements
```
CUDA 11.6
pytorch 1.13.1
tinycudann 
nerfacc
```

## Training
```
python main.py --config-path configs/config.py
```

## Test
```
python main.py --config-path configs/config.py --validate-only
```
