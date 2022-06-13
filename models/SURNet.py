import tensorflow as tf
from keras.models import Input, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D
from keras.layers import add, Lambda, Concatenate, ZeroPadding2D, Reshape, Dense, Permute, multiply, GlobalMaxPooling2D
from keras.optimizers import Adam, SGD

import keras.backend as K
from .lovasz import lovasz_loss
from .losses import *

def surblock(layer_input, skip_input, filters, f_size=3):

    skip = squeeze_excite_block(skip_input)
    up = UpSampling2D(size=2)(layer_input)
    up = squeeze_excite_block(up)
    cat = Concatenate()([up, skip])
    cat = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(cat)
    cat = BatchNormalization(momentum=0.8)(cat) 
    
    fusion = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(cat)
    
    fusion = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(fusion)
    fusion = BatchNormalization(momentum=0.8)(fusion)

    res = add([cat, fusion])
    return res

def myUpSample2X(layer_input, skip_input, filters, f_size=3):
    ## for upsampling
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    u = BatchNormalization(momentum=0.8)(u)
    u = Concatenate()([u, skip_input])
    return u
def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = -1 # if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)


    x = multiply([init, se])
    return x

def resdiual_block(input, num_filters = 16):
    x_1 = BatchNormalization()(input)
    x_1 = Activation('relu')(x_1)
    x_2 = Conv2D(num_filters, kernel_size = 3, strides = 1, padding='same', activation='relu')(input)
    x_2 = Conv2D(num_filters, kernel_size = 3, strides = 1, padding='same', activation='relu')(x_2)
    x = add([x_1, x_2])
    return x


class RUSnet():
    
    
    def __init__(self, im_res=(320, 240, 3), n_classes=5):
        
        def loss_chose():
            from config import args_setting
            args = args_setting()
            if args.loss == 'BCE':
                loss_chosed = 'binary_crossentropy'
            elif args.loss == 'FocalTversky':
                loss_chosed = [FocalTverskyLoss]
            elif args.loss == 'lovaz':
                loss_chosed = [lovasz_loss]
            else:
                loss_chosed = [weightBCE]
            return loss_chosed
        loss_chosed = loss_chose()
        self.lr0 = 1e-4 
        self.inp_shape = (im_res[0], im_res[1])
        self.img_shape = (im_res[0], im_res[1], 3)

        self.model = self.get_model(n_classes)
        self.model.compile(optimizer = Adam(lr = self.lr0), 
                            loss = loss_chosed, metrics = ['accuracy']) # 
        self.model.summary()
    
    def get_model(self, n_classes):

        inputs = Input(self.img_shape)

        # encoder
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        merge4_3 = surblock(pool4, pool3, 256)
        dec1 = surblock(pool4, merge4_3, 512) # 64, 64, 768

        merge3_2 = surblock(pool3, pool2, 128)
        dec2 = surblock(dec1, merge3_2, 256) # 128, 128, 384

        merge2_1 = surblock(pool2, pool1, 64)
        dec3 = surblock(dec2, merge2_1, 128) # 256, 256, 192

        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(inputs, out)

class RUSnet_vgg16():

    def __init__(self, im_res=(320, 240, 3), n_classes=5):
        
        def loss_chose():
            from config import args_setting
            args = args_setting()
            if args.loss == 'BCE':
                loss_chosed = 'binary_crossentropy'
            elif args.loss == 'FocalTversky':
                loss_chosed = [FocalTverskyLoss]
            elif args.loss == 'lovaz':
                loss_chosed = [lovasz_loss]
            elif args.loss == 'focal':
                loss_chosed = [FocalLoss]
            else:
                loss_chosed = [weightBCE]
            return loss_chosed
        loss_chosed = loss_chose()
        self.lr0 = 1e-4 
        self.inp_shape = (im_res[0], im_res[1])
        self.img_shape = (im_res[0], im_res[1], 3)

        self.model = self.get_model_VGG16(n_classes)
        self.model.compile(optimizer = Adam(lr = self.lr0), 
                            loss = loss_chosed, metrics = ['accuracy']) # 
        self.model.summary()
    
    def get_model_VGG16(self, n_classes):
        from keras.applications.vgg16 import VGG16
        vgg = VGG16(input_shape=self.img_shape, include_top=False, weights='imagenet')
        vgg.trainable = True
        for layer in vgg.layers:
            layer.trainable = True
        # encoder
        pool1 = vgg.get_layer('block1_pool').output # 256, 256, 64 
        pool2 = vgg.get_layer('block2_pool').output # 128, 128, 128
        pool3 = vgg.get_layer('block3_pool').output # 64, 64, 256
        pool4 = vgg.get_layer('block4_pool').output # 32, 32, 512

        merge4_3 = surblock(pool4, pool3, 256)
        dec1 = surblock(pool4, merge4_3, 512)

        merge3_2 = surblock(pool3, pool2, 128)
        dec2 = surblock(dec1, merge3_2, 256) # 128, 128, 384

        merge2_1 = surblock(pool2, pool1, 64)
        dec3 = surblock(dec2, merge2_1, 128) # 256, 256, 192

        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(vgg.input, out)

class RUSnet_vgg19():
    def __init__(self, im_res=(320, 240, 3), n_classes=5):
        def loss_chose():
            from config import args_setting
            args = args_setting()
            if args.loss == 'BCE':
                loss_chosed = 'binary_crossentropy'
            elif args.loss == 'FocalTversky':
                loss_chosed = [FocalTverskyLoss]
            elif args.loss == 'lovaz':
                loss_chosed = [lovasz_loss]
            else:
                loss_chosed = [weightBCE]
            return loss_chosed
        loss_chosed = loss_chose()
        self.lr0 = 1e-4 
        self.inp_shape = (im_res[0], im_res[1])
        self.img_shape = (im_res[0], im_res[1], 3)

        self.model = self.get_model_VGG19(n_classes)
        self.model.compile(optimizer = Adam(lr = self.lr0), 
                            loss = loss_chosed, metrics = ['accuracy']) # 
        self.model.summary()
    
    def get_model_VGG19(self, n_classes):
        from keras.applications.vgg19 import VGG19
        vgg = VGG19(input_shape=self.img_shape, include_top=False, weights='imagenet')
        vgg.trainable = True
        for layer in vgg.layers:
            layer.trainable = True
        pool1 = vgg.get_layer('block1_pool').output # 256, 256, 64 
        pool2 = vgg.get_layer('block2_pool').output # 128, 128, 128
        pool3 = vgg.get_layer('block3_pool').output # 64, 64, 256
        pool4 = vgg.get_layer('block4_pool').output # 32, 32, 512

        merge4_3 = surblock(pool4, pool3, 256)
        dec1 = surblock(pool4, merge4_3, 512) # 64, 64, 768

        merge3_2 = surblock(pool3, pool2, 128)
        dec2 = surblock(dec1, merge3_2, 256) # 128, 128, 384

        merge2_1 = surblock(pool2, pool1, 64)
        dec3 = surblock(dec2, merge2_1, 128) # 256, 256, 192


        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(vgg.input, out)
class RUSnet_resnet50():

    def __init__(self, im_res=(320, 240, 3), n_classes=5):
        def loss_chose():
            from config import args_setting
            args = args_setting()
            if args.loss == 'BCE':
                loss_chosed = 'binary_crossentropy'
            elif args.loss == 'FocalTversky':
                loss_chosed = [FocalTverskyLoss]
            elif args.loss == 'lovaz':
                loss_chosed = [lovasz_loss]
            else:
                loss_chosed = [weightBCE]
            return loss_chosed
        loss_chosed = loss_chose()
        self.lr0 = 1e-4 
        self.inp_shape = (im_res[0], im_res[1])
        self.img_shape = (im_res[0], im_res[1], 3)

        self.model = self.get_model_resnet50(n_classes)
        self.model.compile(optimizer = Adam(lr = self.lr0), 
                            loss = loss_chosed, metrics = ['accuracy']) # 
        self.model.summary()
    
    def get_model_resnet50(self, n_classes):
        from keras.applications.resnet50  import ResNet50
        resnet50 = ResNet50(input_shape=self.img_shape, include_top=False, weights='imagenet')
        resnet50.trainable = True
        for layer in resnet50.layers:
            layer.trainable = True

        pool1 = resnet50.get_layer('activation_1').output # 256, 256, 64 
        pool2 = resnet50.get_layer('activation_10').output # 128, 128, 128
        pool3 = resnet50.get_layer('activation_22').output # 64, 64, 256
        pool4 = resnet50.get_layer('activation_40').output # 32, 32, 512

        merge4_3 = surblock(pool4, pool3, 256)
        dec1 = surblock(pool4, merge4_3, 512) # 64, 64, 768

        merge3_2 = surblock(pool3, pool2, 128)
        dec2 = surblock(dec1, merge3_2, 256) # 128, 128, 384

        merge2_1 = surblock(pool2, pool1, 64)
        dec3 = surblock(dec2, merge2_1, 128) # 256, 256, 192



        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(resnet50.input, out)

if __name__=="__main__":
    net = RUSnet_vgg19(im_res=(512, 512, 3))