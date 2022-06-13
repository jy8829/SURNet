import tensorflow as tf
import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def DiceBCELoss(targets, inputs):    
    # print(targets.shape, inputs.shape)
    #flatten label and prediction tensors
    
    BCE =  K.binary_crossentropy(targets, inputs)
    
    Dice_BCE = BCE + (1. - dice_coef(targets, inputs, smooth=1e-6))
    
    return Dice_BCE

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        # BCE =  K.binary_crossentropy(y_true, y_pred)
        Dice_BCE = DiceBCELoss(y_true, y_pred)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        loss = K.mean(focal_loss) + Dice_BCE
        return loss
    return binary_focal_loss_fixed


def weightBCE(target, inputs):
    weights = (target * 59.) + 1.
    bce = K.binary_crossentropy(target, inputs)
    weight_bce = K.mean(bce * weights)
    return weight_bce

def Combo_loss(y_true, y_pred):
    ce_w = 0.5
    ce_d_w = 0.5
    e = K.epsilon()
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
    out = - (ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * d)
    return combo

def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
            
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1 - Tversky), gamma)
    
    return FocalTversky

def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):
        
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
    return 1 - Tversky

def FocalLoss(y_true, y_pred):   
    
    alpha = 0.8
    gamma = 2

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss