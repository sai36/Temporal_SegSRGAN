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

#from tensorflow.python.keras.legacy import interfaces
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
#from tensorflow.python.ops import math_ops, state_ops, control_flow_ops ,array_ops
from tensorflow.python.framework import ops
from .multi import _apply_lr_multiplier
from tensorflow.python.ops import control_flow_ops
class LR_Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., multipliers=None, name='Adam', **kwargs):
        super(LR_Adam, self).__init__(name, **kwargs)
        self.iterations = K.variable(0, name='iterations')
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', decay)
        self.epsilon = epsilon
        self.initial_decay = decay
        self.lr_multipliers = multipliers

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, 'ms') #previous variable i.e. weight or bias
        for var in var_list:
            self.add_slot(var, 'vs')
        self._updates_per_iter = len(var_list) 
    @tf.function
    def _resource_apply_dense(self, grad, var):
        #grads = self.get_gradients(loss, params)
        var_dtype = var.dtype.base_dtype
        #self.updates = [K.update_add(self.iterations, 1)]
        ms = self.get_slot(var, 'ms')
        vs = self.get_slot(var, 'vs')
        local_step = self.iterations + 1
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)

        lr = self.learning_rate
        #print ("lr printing inside ADAM", lr)
        lr_t = lr * K.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        #print ("lr_t printing inside ADAM", lr_t)
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)
        #print ("lr_t printing inside ADAM 2", lr_t)
        #m_t = tf.Variable(tf.identity(ms))
        #v_t = tf.Variable(tf.identity(vs))
        m_t = beta_1_t * ms + (1.0 - beta_1_t) * grad
        v_t = beta_2_t * vs + (1.0 - beta_2_t) * K.square(grad)
        print ("var printing inside ADAM", var)
        var_delta = m_t / (K.sqrt(v_t) + epsilon_t)
        print ("var_delta printing inside ADAM", var_delta)
        var_t = var
        var_t = var - lr_t * var_delta
        var_update = var_t
        var.assign(var_update)
        ms.assign(m_t)
        vs.assign(v_t)
        updates = [var_update, m_t, v_t]
        #return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(LR_Adam, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon})
        return config
