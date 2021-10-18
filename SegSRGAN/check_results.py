
from Function_for_application_test_python3 import segmentation
msg = segmentation(input_file_path = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save/5", step = 20,new_resolution = (0.875, 0.875, 2.5),  path_output_cortex = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save/5", path_output_hr = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save/5", path_output_mask = "/home/a2010-venmo/SRGAN/home_backup/temporal_code_2/mask_SegSRGAN/output_save/5", weights_path = "/proj/SegSRGAN/snapshot_convlstm_conv3d/SegSRGAN_epoch_53", interpolation_type = "NearestNeighbor")
#msg = segmentation(input_file_path = "/home/a2010-venmo/low_res_cropped_303_new.nii", step = 20,new_resolution = (0.875, 0.875, 2.5),  path_output_cortex = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/cortex_output/cortex_303_101.nii", path_output_hr = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/hr_output/hr_303_101.nii", path_output_mask = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/mask_output/mask_303_101.nii", weights_path = "/proj/SegSRGAN/snapshot_cropped_full/SegSRGAN_epoch_101", interpolation_type = "NearestNeighbor")
#msg = segmentation(input_file_path = "/home/a2010-venmo/low_res_cropped_303_new.nii", step = 20,new_resolution = (0.875, 0.875, 2.5),  path_output_cortex = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/cortex_output/cortex_303_169.nii", path_output_hr = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/hr_output/hr_303_169.nii", path_output_mask = "/home/a2010-venmo/mask_SegSRGAN/cropped_full/mask_output/mask_303_169.nii", weights_path = "/proj/SegSRGAN/snapshot_cropped_full/SegSRGAN_epoch_169", interpolation_type = "NearestNeighbor")


print (msg)

