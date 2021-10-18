This directory is for SegSRGAN without LSTM. The input is a minibatch consisting of multiple patches generated from 3D image. The output is also set of patches. <br />

The code for running the training is given by: 
'''python SegSRGAN_training_modified_full.py -n 1.75 1.75 5 -csv /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/temporal_cropped_dgx_1.csv -sf /proj/SegSRGAN/snapshot_baseline_temporal/ -dice /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/dice_files/dice_baseline_temporal.csv -mse /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/mse_files/mse_baseline_temporal.csv -folder_training_data /proj/SegSRGAN/temp_training_temporal -e 200 -b 64 -rl dice '''
