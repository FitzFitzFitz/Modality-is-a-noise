python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=20122 --use_env train_mm.py --cfg ~/DELIVER/configs/mcubes_rgbadn.yaml
