"""
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

"""
This code is modified and for handling temporal data. Time Distributed functions
are used for all the layers expect for the LSTM or ConvLSTM layers. The input
shape is 6D tensor.
"""

import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Reshape
from tensorflow.keras.layers import Conv3D, Add, UpSampling3D, Activation, Concatenate, ConvLSTM3D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.layers import TimeDistributed, Dropout
from tensorflow.keras.initializers import lecun_normal
import tensorflow.keras.backend as K
import os
from .Adam_lr_mult import LR_Adam

gen_initializer = lecun_normal()
strategy = tf.distribute.MirroredStrategy(['GPU:0'])
from .layers import  ReflectPadding3D, InstanceNormalization3D, \
    activation_SegSRGAN
from .loss import wasserstein_loss,gradient_penalty_loss,Reconstruction_loss_function,Dice_loss
#from .Adam_lr_mult import LR_Adam
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
#from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def resnet_blocks(input_res, kernel, name):
    in_res_1 = TimeDistributed(ReflectPadding3D(padding=1))(input_res)
    out_res_1 = ConvLSTM3D(kernel, 3, strides=1, kernel_initializer=gen_initializer,
                       use_bias=False,activation = None,
                       name=name + '_conv_a',
                       data_format='channels_first', return_sequences=True)(in_res_1)
    out_res_1 = TimeDistributed(InstanceNormalization3D(name=name + '_isnorm_a'), name=name + '_isnorm_a')(out_res_1)
    out_res_1 = TimeDistributed(Activation('relu'))(out_res_1)
    #print ("IN RESNET BLOCK, AFTER ACTIVATION CONV_A", out_res_1)
    in_res_2 = TimeDistributed(ReflectPadding3D(padding=1))(out_res_1)
    out_res_2 = ConvLSTM3D(kernel, 3, strides=1, kernel_initializer=gen_initializer,
                       use_bias=False, activation = None,
                       name=name + '_conv_b',
                       data_format='channels_first', return_sequences=True)(in_res_2)
    out_res_2 = TimeDistributed(InstanceNormalization3D(name=name + '_isnorm_b'), name=name + '_isnorm_b')(out_res_2)

    out_res = TimeDistributed(Add())([out_res_2, input_res])
    return out_res


class SegSRGAN(object):
    """Description of the GAN network structure"""
    def __init__(self, u_net_gen, temp_inp = 10, image_row=64, image_column=64, image_depth=64,
                 first_discriminator_kernel=32, first_generator_kernel=16,
                 lamb_rec=1, lamb_adv=0.001, lamb_gp=10,
                 lr_dis_model=0.0001, lr_gen_model=0.0001, multi_gpu=True,
                 is_conditional=False,
                 is_residual=True,
                 fit_mask = False,
                 nb_classe_mask = 0,
                 loss_name="charbonnier"):

        if (image_row %4!=0) |  (image_column %4!=0) | (image_depth %4!=0) :

            raise AssertionError('Patch size must be divisible by 4')

        self.image_row = image_row
        self.image_column = image_column
        self.image_depth = image_depth
        self.D = None  # discriminator
        self.G = None  # generator
        self.dis_model = None  # discriminator model
        self.dis_model_multi_gpu = None
        self.gen_model = None  # generator model
        self.gen_model_multi_gpu = None
        self.discriminator_kernel = first_discriminator_kernel  # profondeur des carac extraite pour le gen
        self.generator_kernel = first_generator_kernel  # profondeur des carac extraites pour le discri
        self.lamb_adv = lamb_adv
        self.lamb_rec = lamb_rec
        self.lamb_gp = lamb_gp
        self.lr_dis_model = lr_dis_model
        self.lr_gen_model = lr_gen_model
        self.u_net_gen = u_net_gen
        self.multi_gpu = multi_gpu
        self.is_conditional = is_conditional
        self.is_residual = is_residual
        self.fit_mask = fit_mask
        self.nb_classe_mask = nb_classe_mask
        self.loss_name = loss_name

    def discriminator_block(self, name):
        """Creates a discriminator model that takes an image as input and outputs a single value, representing whether
        the input is real or generated. Unlike normal GANs, the output is not sigmoid and does not represent a
        probability!
        Instead, the output should be as large and negative as possible for generated inputs and as large and positive
        as possible for real inputs.
        Note that the improved WGAN paper suggests that BatchNormalization should not be used in the discriminator."""
        with strategy.scope():
            if self.fit_mask :

                inputs = Input(shape=(2+self.nb_classe_mask, self.image_row, self.image_column, self.image_depth), name='dis_input')
            else :
                # In:
                inputs = Input(shape=(10, 2, self.image_row, self.image_column, self.image_depth), name='dis_input')

            # Input 64
            disnet = TimeDistributed(Conv3D(self.discriminator_kernel * 1, 4, strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            data_format='channels_first',
                            name=name + '_conv_dis_1'),  name=name + '_conv_dis_1')(inputs)
            disnet = TimeDistributed(BatchNormalization(), name = name + 'batch_1')(disnet)
            disnet = TimeDistributed(LeakyReLU(0.01))(disnet)
            disnet = TimeDistributed(Dropout(0.25))(disnet)

            # Hidden 1 : 32
            disnet = TimeDistributed(Conv3D(self.discriminator_kernel * 2, 4, strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            data_format='channels_first',
                            name=name + '_conv_dis_2'), name=name + '_conv_dis_2')(disnet)
            disnet = TimeDistributed(BatchNormalization(),  name = name + 'batch_2')(disnet)
            disnet = TimeDistributed(LeakyReLU(0.01))(disnet)
            disnet = TimeDistributed(Dropout(0.25))(disnet)

            # Hidden 2 : 16
            disnet = TimeDistributed(Conv3D(self.discriminator_kernel * 4, 4, strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            data_format='channels_first',
                            name=name + '_conv_dis_3'), name=name + '_conv_dis_3')(disnet)
            disnet = TimeDistributed(BatchNormalization(), name = name + 'batch_3')(disnet)
            disnet = TimeDistributed(LeakyReLU(0.01))(disnet)
            disnet = TimeDistributed(Dropout(0.25))(disnet)

            # Hidden 3 : 8
            disnet = TimeDistributed(Conv3D(self.discriminator_kernel * 8, 4, strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            data_format='channels_first',
                            name=name + '_conv_dis_4'), name=name + '_conv_dis_4')(disnet)
            disnet = TimeDistributed(BatchNormalization(), name = name + 'batch_4')(disnet)
            disnet = TimeDistributed(LeakyReLU(0.01))(disnet)
            disnet = TimeDistributed(Dropout(0.25))(disnet)

            # Hidden 4 : 4
            disnet = TimeDistributed(Conv3D(self.discriminator_kernel * 16, 4, strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            data_format='channels_first',
                            name=name + '_conv_dis_5'), name=name + '_conv_dis_5')(disnet)
            disnet = TimeDistributed(BatchNormalization(), name = name + 'batch_5')(disnet)
            disnet = TimeDistributed(LeakyReLU(0.01))(disnet)
            disnet = TimeDistributed(Dropout(0.25))(disnet)

            # Decision : 2
            decision = TimeDistributed(Conv3D(1, 2, strides=1,
                              use_bias=False,
                              kernel_initializer='he_normal',
                              data_format='channels_first',
                              name='dis_decision'), name='dis_decision')(disnet)

            decision = TimeDistributed(Reshape((1,)))(decision)

            model = Model(inputs=[inputs], outputs=[decision], name=name)
            return model

    def generator_block(self, name):
        """Creates a generator model"""
        with strategy.scope():
            # generator same dim as input and output if multiple of 4
            inputs = Input(shape=(10, 1, self.image_row, self.image_column, self.image_depth))

            # Representation
            gennet = TimeDistributed(ReflectPadding3D(padding=3))(inputs)
            #print ("AFTER 1st REFLECT PADDING", gennet)
            gennet = TimeDistributed(Conv3D(self.generator_kernel, 7, strides=1, kernel_initializer=gen_initializer,
                            use_bias=False,
                            name=name + '_gen_conv1',
                            data_format='channels_first'), name=name + '_gen_conv1')(gennet)
            #print ("TIME DIST CONV", gennet)
            gennet = TimeDistributed(InstanceNormalization3D(name=name + '_gen_isnorm_conv1'), name=name + '_gen_isnorm_conv1')(gennet)
            #print ("INSTANCE NORMALIZATION", gennet)
            gennet = TimeDistributed(Activation('relu'))(gennet)
            #print ("ACTIVATION", gennet)

            # Downsampling 1
            gennet = TimeDistributed(ReflectPadding3D(padding=1))(gennet)
            gennet = TimeDistributed(Conv3D(self.generator_kernel * 2, 3, strides=2, kernel_initializer=gen_initializer,
                            use_bias=False,
                            name=name + '_gen_conv2',
                            data_format='channels_first'), name=name + '_gen_conv2')(gennet)
            gennet = TimeDistributed(InstanceNormalization3D(name=name + '_gen_isnorm_conv2'), name=name + '_gen_isnorm_conv2')(gennet)
            gennet = TimeDistributed(Activation('relu'))(gennet)

            # Downsampling 2
            gennet = TimeDistributed(ReflectPadding3D(padding=1))(gennet)
            gennet = TimeDistributed(Conv3D(self.generator_kernel * 4, 3, strides=2, kernel_initializer=gen_initializer,
                            use_bias=False,
                            name=name + '_gen_conv3',
                            data_format='channels_first'), name=name + '_gen_conv3')(gennet)
            gennet = TimeDistributed(InstanceNormalization3D(name=name + '_gen_isnorm_conv3'), name=name + '_gen_isnorm_conv3')(gennet)
            gennet = TimeDistributed(Activation('relu'))(gennet)
            #print (gennet)

            # Resnet blocks : 6, 8*4 = 32
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block1')
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block2')
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block3')
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block4')
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block5')
            gennet = resnet_blocks(gennet, self.generator_kernel * 4, name=name + '_gen_block6')

            # Upsampling 1
            gennet = TimeDistributed(UpSampling3D(size=(2, 2, 2),
                                  data_format='channels_first'))(gennet)
            #print ("AFTER UPSAMPLING 1", gennet)
            gennet = TimeDistributed(ReflectPadding3D(padding=1))(gennet)
            gennet = TimeDistributed(Conv3D(self.generator_kernel * 2, 3, strides=1, kernel_initializer=gen_initializer,
                            use_bias=False,
                            name=name + '_gen_deconv1',
                            data_format='channels_first'), name=name + '_gen_deconv1')(gennet)
            gennet = TimeDistributed(InstanceNormalization3D(name=name + '_gen_isnorm_deconv1'), name=name + '_gen_isnorm_deconv1')(gennet)
            gennet = TimeDistributed(Activation('relu'))(gennet)

            # Upsampling 2
            gennet = TimeDistributed(UpSampling3D(size=(2, 2, 2),
                                  data_format='channels_first'))(gennet)
            gennet = TimeDistributed(ReflectPadding3D(padding=1))(gennet)
            gennet = TimeDistributed(Conv3D(self.generator_kernel, 3, strides=1, kernel_initializer=gen_initializer,
                            use_bias=False,
                            name=name + '_gen_deconv2',
                            data_format='channels_first'), name=name + '_gen_deconv2')(gennet)
            gennet = TimeDistributed(InstanceNormalization3D(name=name + '_gen_isnorm_deconv2'), name=name + '_gen_isnorm_deconv2')(gennet)
            gennet = TimeDistributed(Activation('relu'))(gennet)

            # Reconstruction
            gennet = TimeDistributed(ReflectPadding3D(padding=3))(gennet)
            if self.fit_mask :
                gennet = Conv3D(2+self.nb_classe_mask, 7, strides=1, kernel_initializer=gen_initializer,
                                use_bias=False,
                                name=name + '_gen_1conv',
                                data_format='channels_first')(gennet)
            else :
                gennet = TimeDistributed(Conv3D(2, 7, strides=1, kernel_initializer=gen_initializer,
                                use_bias=False,
                                name=name + '_gen_1conv',
                                data_format='channels_first'), name=name + '_gen_1conv')(gennet)

            predictions = gennet
            predictions = TimeDistributed(activation_SegSRGAN(is_residual=self.is_residual,nb_classe_mask=self.nb_classe_mask,fit_mask=self.fit_mask))(
                [predictions, inputs])  # sigmoid proba + add input and pred SR

            model = Model(inputs=inputs, outputs=predictions, name=name)
            return model
    def generator(self):
        with strategy.scope():
            if self.G:
                return self.G

            if self.is_residual :

                residual_string = ""

            else :

                residual_string = "_nn_residual"

            if self.u_net_gen  :

                if self.is_conditional :

                    self.G = self.generator_block_u_net_cond('G_unet_cond'+residual_string)

                else :

                    self.G = self.generator_block_u_net('G_unet'+residual_string)

            else :

                if self.is_conditional :

                    self.G = self.G = self.generator_block_conditionnal('G_cond'+residual_string)

                else :

                    self.G = self.generator_block('G'+residual_string)

            return self.G


    def discriminator(self):
        with strategy.scope():
            if self.D:
                return self.D
            if self.is_conditional:
                self.D = self.discriminator_block_conditionnal('DX_cond')
            else:
                self.D = self.discriminator_block('DX')
            return self.D

    def generator_multi_gpu(self):
        with strategy.scope():
            num_gpu = len(get_available_gpus())

            if self.multi_gpu and (num_gpu > 1):

                gen_multi_gpu = multi_gpu_model(self.generator(), gpus=num_gpu, cpu_merge=False)
                print("Generator duplicated on : " + str(num_gpu) + " GPUS")
            else:
                gen_multi_gpu = self.generator()

            return gen_multi_gpu

    def discri_multi_gpu(self):
        with strategy.scope():
            num_gpu = len(get_available_gpus())

            if self.multi_gpu and (num_gpu > 1):

                discri_multi_gpu = multi_gpu_model(self.discriminator(), gpus=num_gpu, cpu_merge=False)
                print("Generator duplicated on : " + str(num_gpu) + " GPUS")
            else:
                discri_multi_gpu = self.discriminator()

            return discri_multi_gpu

    def generator_model(self):

        # We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !
        #        all_parameters = 63
        #        generator_parameters = 52
        if self.gen_model:
            return self.gen_model
        with strategy.scope():
            #if self.gen_model:
                #return self.gen_model
            print("We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !")
            all_parameters = len(self.generator().get_weights()) + len(self.discriminator().get_weights())
            generator_parameters = len(self.generator().get_weights())
            multipliers = {'gen': 1, 'DX': 0}
            '''multipliers = np.ones(all_parameters)
            for idx in range(generator_parameters, all_parameters):
                multipliers[idx] = 0.0'''

            input_im = Input(shape=(10, 1, self.image_row, self.image_column, self.image_depth), name='input_im_gen')

            if self.is_conditional:

                input_res = Input(shape=(1, self.image_row, self.image_column, self.image_depth), name='input_res_gen')
                gx_gen = self.generator()([input_im, input_res])  # Fake X
                fool_decision = self.discriminator()([gx_gen, input_res])  # Fooling D
                # Model
                self.gen_model = Model([input_im, input_res], [fool_decision, gx_gen])

            else:
                gx_gen = self.generator()(input_im)  # Fake X
                fool_decision = self.discriminator()(gx_gen)  # Fooling D
                # Model
                self.gen_model = Model(input_im, [fool_decision, gx_gen])

            # print archi :

            self.generator().summary(line_length=150)

            num_gpu = len(get_available_gpus())
            '''
            if (self.multi_gpu) & (num_gpu > 1):

                self.gen_model_multi_gpu = multi_gpu_model(self.gen_model, gpus=num_gpu, cpu_merge=False)
                print("Generator Model duplicated on : " + str(num_gpu) + " GPUS")
            else:
                self.gen_model_multi_gpu = self.gen_model
                print("Generator Model apply on CPU or single GPU")'''
            self.gen_model_multi_gpu = self.gen_model
            #print(self.loss_name)

            reconstruction_loss = Reconstruction_loss_function(self.loss_name).get_loss_function()


                # self.gen_model = multi_gpu_model(self.gen_model, gpus=num_gpu)
            self.gen_model_multi_gpu.compile(optimizer=LR_Adam(learning_rate=self.lr_gen_model, beta_1=0.5, beta_2=0.999, multipliers = multipliers),
                                            loss=[wasserstein_loss, reconstruction_loss],
                                            loss_weights=[self.lamb_adv, self.lamb_rec], experimental_run_tf_function=False)

            return self.gen_model, self.gen_model_multi_gpu

    def generator_model_for_pred(self):
        if (1):

            if self.gen_model:
                return self.gen_model

            print("We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !")
            # We freeze the weights of Discriminator by setting their learning rate as 0 when updating Generator !
            all_parameters = 63
            generator_parameters = 52
            multipliers = {'gen': 1, 'DX': 0}
            '''multipliers = np.ones(all_parameters)
            for idx in range(generator_parameters, all_parameters):
                multipliers[idx] = 0.0'''

            # Input
            input_gen = Input(shape=(10, 1, self.image_row, self.image_column, self.image_depth), name='input_gen')

            if self.is_conditional:

                input_res = Input(shape=(1, self.image_row, self.image_column, self.image_depth), name='input_res_gen')
                gx_gen = self.generator()([input_gen, input_res])  # Fake X
                # Model
                self.gen_model = Model([input_gen, input_res], [gx_gen])

            else:

                gx_gen = self.generator()(input_gen)  # Fake X
                # Model
                self.gen_model = Model(input_gen, [gx_gen])

            # ajout d'un loss quelconque car nous en avons pas besoin pour l'application d'un model mais il ne faut pas que
            # celui-ci soit definit
            # en utilisant de discri
            self.gen_model.compile(optimizer=LR_Adam(learning_rate=self.lr_gen_model, beta_1=0.5, beta_2=0.999, multipliers=multipliers),
                                    loss=losses.mean_squared_error, experimental_run_tf_function=False)

            return self.gen_model

    def discriminator_model(self):
        with strategy.scope():
            if self.dis_model:
                return self.dis_model

            if self.fit_mask:

                # Input
                real_dis = Input(shape=(2+self.nb_classe_mask, self.image_row, self.image_column, self.image_depth), name='real_dis')
                fake_dis = Input(shape=(2+self.nb_classe_mask, self.image_row, self.image_column, self.image_depth), name='fake_dis')
                interp_dis = Input(shape=(2+self.nb_classe_mask, self.image_row, self.image_column, self.image_depth), name='interp_dis')

            else :

                # Input
                real_dis = Input(shape=(10, 2, self.image_row, self.image_column, self.image_depth), name='real_dis')
                fake_dis = Input(shape=(10, 2, self.image_row, self.image_column, self.image_depth), name='fake_dis')
                #interp_dis = Input(shape=(10, 2, self.image_row, self.image_column, self.image_depth), name='interp_dis')


            if self.is_conditional:

                res = Input(shape=(1, self.image_row, self.image_column, self.image_depth))

                # Discriminator
                real_decision = self.discriminator()([real_dis, res])  # Real X
                fake_decision = self.discriminator()([fake_dis, res])  # Fake X
                interp_decision = self.discriminator()([interp_dis, res])  # interpolation X

            else:

                # Discriminator
                real_decision = self.discriminator()(real_dis)  # Real X
                fake_decision = self.discriminator()(fake_dis)  # Fake X
                #interp_decision = self.discriminator()(interp_dis)  # interpolation X

            # GP loss
            '''partial_gp_loss = partial(gradient_penalty_loss,
                                      averaged_samples=interp_dis)
            partial_gp_loss.__name__ = 'gradient_penalty' '''  # Functions need names or Keras will throw an error*

            num_gpu = len(get_available_gpus())

            print("number of gpus : " + str(num_gpu))
            if self.is_conditional:

                # Model
                self.dis_model = Model([real_dis, fake_dis, interp_dis, res],
                                      [real_decision, fake_decision, interp_decision])
            else:
                # Model
                self.dis_model = Model([real_dis, fake_dis], [real_decision, fake_decision])

            '''if (self.multi_gpu) & (num_gpu > 1):

                self.dis_model_multi_gpu = multi_gpu_model(self.dis_model, gpus=num_gpu, cpu_merge=False)
                print("Discriminator Model duplicated on : " + str(num_gpu) + " GPUS")
            else:
                self.dis_model_multi_gpu = self.dis_model
                print("Discriminator Model apply on CPU or single GPU")'''
            self.dis_model_multi_gpu = self.dis_model
            # self.dis_model = multi_gpu_model(self.dis_model, gpus=num_gpu)
            self.dis_model_multi_gpu.compile(optimizer=Adam(learning_rate=self.lr_dis_model, beta_1=0.5, beta_2=0.999),
                                            loss=[wasserstein_loss, wasserstein_loss],
                                            loss_weights=[1, 1], experimental_run_tf_function=False)
            # multi gpu training ne change rien au temps d'exectution sur romeo. meme en changeant l'argument cpu_merge=False.

            return self.dis_model, self.dis_model_multi_gpu
