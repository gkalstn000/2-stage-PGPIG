nohup python train.py --id base_cycle_ptm_spade --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 14 --num_workers 15 --gan_mode wgangp --continue_train > base_cycle_ptm_spade.out &

nohup python train.py --id base_cycle_ptm_spade_nofst --tf_log --gpu_ids 3 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 14 --num_workers 15 --gan_mode wgangp > base_cycle_ptm_spade_nofst.out &

nohup python train.py --id z_cycle_ptm_spade --tf_log --gpu_ids 2 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 26 --num_workers 15 --continue_train > z_cycle_ptm_spade.out &
nohup python train.py --id VAE_spade_full --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 20 --num_workers 15  --continue_train > VAE_spade_full.out &

nohup python train.py --id VAE_DPTN --tf_log --gpu_ids 0 --type_En_c attn --type_Dc spade --pose_nc 18 --batchSize 15 --num_workers 15  > VAE_DPTN.out &


# Red
nohup python train.py --id spade_light --tf_log --gpu_ids 2 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 15 --dataroot /datasets/msha/fashion > spade_light.out &
nohup python train.py --id spade_128 --tf_log --gpu_ids 1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 16 --num_workers 8 --z_dim 128 --continue_train > spade_128.out &
nohup python train.py --id spade_192 --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 16 --num_workers 8 --z_dim 192 --continue_train > spade_192.out &
nohup python train.py --id spain_256 --tf_log --gpu_ids 2 --type_En_c z --type_Dc spain --pose_nc 41 --batchSize 12 --num_workers 5  --z_dim 256 --continue_train > spain_256.out &





# In NIPA
nohup python train.py --id spade_d --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 35 --num_workers 7 --dataroot /home/work/msha/datasets/fashion > spade_d.out &
nohup python train.py --id spade_origin --tf_log --gpu_ids 1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 4 --num_workers 15 --dataroot /home/work/msha/datasets/fashion --continue_train > spade_origin.out &
nohup python train.py --id spade_origin_pair --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 15 --dataroot /home/work/msha/datasets/fashion > spade_origin_pair.out &


nohup python train.py --id spade_256 --tf_log --gpu_ids 0 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 7 --z_dim 256 --dataroot /home/work/msha/datasets/fashion --continue_train > spade_256.out &
nohup python train.py --id spade_512 --tf_log --gpu_ids 1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 7 --z_dim 512 --dataroot /home/work/msha/datasets/fashion --continue_train > spade_512.out &
nohup python train.py --id spade_256_zloss --tf_log --gpu_ids 1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 20 --num_workers 7 --z_dim 256 --dataroot /home/work/msha/datasets/fashion --continue_train> spade_256_zloss.out &

nohup python train.py --id spade_256_syncbatch --tf_log --gpu_ids 0,1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 40 --num_workers 15 --z_dim 256 --dataroot /home/work/msha/datasets/fashion --norm_E spectralsyncbatch --norm_D spectralsyncbatch --norm_G spectralsyncbatch --continue_train > spade_256_syncbatch.out &


nohup python train.py --id spade_128_syncbatch --tf_log --gpu_ids 0,1 --type_En_c z --type_Dc spade --pose_nc 41 --batchSize 36 --num_workers 18 --z_dim 128 --dataroot /home/work/msha/datasets/fashion > spade_128_syncbatch.out &

