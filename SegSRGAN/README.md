This directory is for SegSRGAN without LSTM. The input is a minibatch consisting of multiple patches generated from 3D image. The output is also set of patches. <br />

An example of how to run the code for training is given by:  <br />

```

python SegSRGAN_training_modified_full.py -n 1.75 1.75 5 
-csv /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/temporal_cropped_dgx_1.csv 
-sf /proj/SegSRGAN/snapshot_baseline_temporal/ -dice /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/dice_files/dice_baseline_temporal.csv -mse /home/a2010-venmo/SRGAN/home_backup/mask_SegSRGAN/mse_files/mse_baseline_temporal.csv 
-folder_training_data /proj/SegSRGAN/temp_training_temporal -e 200 -b 64 -rl dice 
```

> * **csv** (string): CSV file that contains the paths to the files used for the training. These files are divided into two categories: train and test. Consequently, it must contain 3 columns, called: HR_image, Label_image and Base (which is equal to either Train or Test), respectively
> * **dice_file** (string): CSV file where to store the DICE at each epoch
> * **mse\_file**(string): CSV file where to store the MSE at each epoch
> * **epoch** (integer) : number of training epochs
> * **batch_size** (integer) : number of patches per mini batch
> * **number\_of\_disciminator\_iteration** (integer): how many times we train the discriminator before training the generator
> * **new_low_res** (tuple): resolution of the LR image generated during the training. One value is given per dimension, for fixed resolution (e.g.“−−new_low_res 0.5 0.5 3”). Two values are given per dimension if the resolutions have to be drawn between bounds (e.g. “−−new_low_res 0.5 0.5 4 −−new_low_res 1 1 2” means that for each image at each epoch, x and y resolutions are uniformly drawn between 0.5 and 1, whereas z resolution is uniformly drawn between 2 and 4.
> * **snapshot_folder** (string): path of the folder in which the weights will be regularly saved after a given number of epochs (this number is given by **snapshot** (integer) argument). But it is also possible to continue a training from saved weights (detailed below).
> * **folder_training_data** (string): folder where temporary files are written during the training (created at the begining of each epoch and deleted at the end of it)
> * **rl**: reconstruction loss. Dice is is showing better performance than the charbonnier loss. 
> * **interp** (string): Interpolation type which is used for the reconstruction of the high resolution image before 
>applying the neural network. Can be either 'scipy' or 'sitk' ('scipy' by default). The downsampling method associated to each 
>interpolation method is different. With Scipy, the downsampling is performed by a Scipy method whereas we perform a classical,
>manual downsampling for sitk. 

