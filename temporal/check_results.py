from Function_for_application_test_python3 import segmentation
msg = segmentation(input_file_path = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save1/5", step = 20,new_resolution = (0.875, 0.875, 2.5),  path_output_cortex = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save1/5", path_output_hr = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save1/5", path_output_mask = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save1/5", weights_path = "/proj/SegSRGAN/snapshot_convlstm_both_GAN2/SegSRGAN_epoch_70", interpolation_type = "NearestNeighbor")



print (msg)

