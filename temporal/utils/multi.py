import tensorflow as tf

def _apply_lr_multiplier(self, lr_t, var):
    
    multiplier_name = [mult_name for mult_name in self.lr_multipliers if mult_name in var.name]
    if multiplier_name != []:
        #print ("INSIDE MULTI.PY NAME = ",multiplier_name)
        #print ("LAYER NAME", var.name)
        lr_mult = self.lr_multipliers[multiplier_name[0]]
    else:
        #print ("INSIDE MULTI.PY NAME NOT FOUND")
        #print ("LAYER NAME", var.name)
        lr_mult = 0
    lr_t = lr_t * lr_mult
    '''
    if self.init_verbose and not self._init_notified:
        lr_print = self._init_lr * lr_mult
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
            '%.e' % round(lr_print, 5), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(
                var.name, lr_print))'''
    return lr_t
