nohup python train.py --id base_cycle_ptm_spade --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 14 --num_workers 15 --gan_mode wgangp --continue_train > base_cycle_ptm_spade.out &

nohup python train.py --id base_cycle_ptm_spade_nofst --tf_log --gpu_ids 3 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 14 --num_workers 15 --gan_mode wgangp > base_cycle_ptm_spade_nofst.out &

nohup python train.py --id z_cycle_ptm_spade --tf_log --gpu_ids 2 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 26 --num_workers 15 --continue_train > z_cycle_ptm_spade.out &
nohup python train.py --id VAE_spade_full --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 20 --num_workers 15  --continue_train > VAE_spade_full.out &

nohup python train.py --id VAE_DPTN --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 15 --num_workers 15  > VAE_DPTN.out &


# Red
nohup python train.py --id spade_light --tf_log --gpu_ids 2 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 15 --dataroot /datasets/msha/fashion > spade_light.out & # stop at 0422
nohup python train.py --id VAE_CF --tf_log --gpu_ids 1 --type_En_c noise --type_Dc default --pose_nc 41 --batchSize 10 --num_workers 20 --dataroot /datasets/msha/fashion --continue_train> VAE_CF.out &
nohup python train.py --id SPAIN --tf_log --gpu_ids 2 --type_En_c noise --type_Dc default --pose_nc 41 --batchSize 30 --num_workers 20 --dataroot /datasets/msha/fashion > SPAIN.out &


# In NIPA
nohup python train.py --id spade_d --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 35 --num_workers 7 --dataroot /home/work/msha/datasets/fashion > spade_d.out &
nohup python train.py --id spade_origin --tf_log --gpu_ids 1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 4 --num_workers 15 --dataroot /home/work/msha/datasets/fashion --continue_train > spade_origin.out &
nohup python train.py --id spade_origin_pair --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 4 --num_workers 15 --dataroot /home/work/msha/datasets/fashion --z_dim 512 --continue_train > spade_origin_pair.out &