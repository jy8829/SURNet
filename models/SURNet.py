import tensorflow as tf
# import keras
# print(keras.__version_)
from keras.models import Input, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, DepthwiseConv2D, AveragePooling2D
from keras.layers import add, Lambda, Concatenate, ZeroPadding2D, Reshape, Dense, Permute, multiply, GlobalMaxPooling2D
from keras.optimizers import Adam, SGD

import keras.backend as K
from .lovasz import lovasz_loss
from .losses import *

def UpSample(layer_input, skip_input, filters, f_size=3):
    ## for upsampling
    skip = squeeze_excite_block(skip_input)
    up = UpSampling2D(size=2)(layer_input)
    up = squeeze_excite_block(up)
    cat = Concatenate()([up, skip])
    # cat = BatchNormalization(momentum=0.8)(cat)
    cat = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(cat)
    cat = BatchNormalization(momentum=0.8)(cat) ###
    # cat = squeeze_excite_block(cat)
    fusion = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(cat)
    # fusion = BatchNormalization(momentum=0.8)(fusion)  ###
    fusion = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(fusion)
    fusion = BatchNormalization(momentum=0.8)(fusion)

    res = add([cat, fusion])
    # res = BatchNormalization(momentum=0.8)(res)
    # res = BatchNormalization(momentum=0.8)(res)
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

    # if K.image_data_format() == 'channels_first':
    # se = Permute((3, 1, 2))(se)

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
    
    """ 
       The SUIM-Net model (Fig. 5 in the paper)
        - base = 'RSB' for RSB-based encoder (Fig. 5b)
        - base = 'VGG' for 12-layer VGG-16 encoder (Fig. 5c)
    """
    

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
        # +++++++++++ (?, 120, 160, 64)
        # +++++++++++ (?, 60, 80, 128)
        # +++++++++++ (?, 30, 40, 256)
        # +++++++++++ (?, 15, 20, 512)
        ## decoder
        # add se block ->v1_se
        # pool3_se = squeeze_excite_block(pool3)  

        # pool4_up = UpSampling2D(size=2)(pool4) 
        # merge4_3 = Concatenate()([pool3_se, pool4_up])
        # merge4_3 = Conv2D(256, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge4_3)
        # merge4_3 = BatchNormalization(momentum=0.8)(merge4_3)
        # # print('-------------------------', merge4_3.shape)
        # merge4_3 = resdiual_block(merge4_3, num_filters = 256)

        merge4_3 = UpSample(pool4, pool3, 256)
        # merge4_3 = squeeze_excite_block(merge4_3)
        dec1 = UpSample(pool4, merge4_3, 512) # 64, 64, 768
        # dec1 = squeeze_excite_block(dec1)
        # print('----------', dec1.shape)

        # add se block ->v1_se
        # pool2_se = squeeze_excite_block(pool2)
        
        # pool3_up = UpSampling2D(size=2)(pool3) 
        # merge3_2 = Concatenate()([pool2_se, pool3_up])
        # merge3_2 = Conv2D(128, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge3_2)
        # merge3_2 = BatchNormalization(momentum=0.8)(merge3_2)  
        # merge3_2 = resdiual_block(merge3_2, num_filters =128)
        merge3_2 = UpSample(pool3, pool2, 128)
        # merge3_2 = squeeze_excite_block(merge3_2)
        dec2 = UpSample(dec1, merge3_2, 256) # 128, 128, 384
        # dec2 = squeeze_excite_block(dec2)
        # print('----------', dec2.shape)

        # add se block ->v1_se
        # pool1_se = squeeze_excite_block(pool1)
        
        # pool2_up = UpSampling2D(size=2)(pool2) 
        # merge2_1 = Concatenate()([pool1_se, pool2_up])
        # merge2_1 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge2_1)
        # merge2_1 = BatchNormalization(momentum=0.8)(merge2_1)  
        # merge2_1 = resdiual_block(merge2_1, num_filters = 64)
        merge2_1 = UpSample(pool2, pool1, 64)
        # merge2_1 = squeeze_excite_block(merge2_1)
        dec3 = UpSample(dec2, merge2_1, 128) # 256, 256, 192
        # dec3 = squeeze_excite_block(dec3)
        # print('----------', dec3.shape)


        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
       
        # dec4 = squeeze_excite_block(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(2, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # print('----------', dec4.shape)
        

        ## return output layer
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(inputs, out)

class RUSnet_vgg16():
    
    """ 
       The SUIM-Net model (Fig. 5 in the paper)
        - base = 'RSB' for RSB-based encoder (Fig. 5b)
        - base = 'VGG' for 12-layer VGG-16 encoder (Fig. 5c)
    """
    

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
        # print('+++++++++++', pool1.shape)
        pool2 = vgg.get_layer('block2_pool').output # 128, 128, 128
        # print('+++++++++++', pool2.shape)
        pool3 = vgg.get_layer('block3_pool').output # 64, 64, 256
        # print('+++++++++++', pool3.shape)
        pool4 = vgg.get_layer('block4_pool').output # 32, 32, 512
        # print('+++++++++++', pool4.shape)
        # print('+++++++++++', pool4.shape)
        ## decoder
        # # add se block ->v1_se
        # pool3_se = squeeze_excite_block(pool3)  

        # pool4_up = UpSampling2D(size=2)(pool4) 
        # merge4_3 = Concatenate()([pool3_se, pool4_up])
        # merge4_3 = Conv2D(256, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge4_3)
        # merge4_3 = BatchNormalization(momentum=0.8)(merge4_3)
        # # print('-------------------------', merge4_3.shape)
        # merge4_3 = resdiual_block(merge4_3, num_filters = 256)
        # dec1 = myUpSample2X(pool4, merge4_3, 512) # 64, 64, 768


        merge4_3 = UpSample(pool4, pool3, 256)
        # merge4_3 = squeeze_excite_block(merge4_3)
        dec1 = UpSample(pool4, merge4_3, 512)
        # dec1 = squeeze_excite_block(dec1)
        # print('----------', dec1.shape)

        # add se block ->v1_se
        # pool2_se = squeeze_excite_block(pool2)
        
        # pool3_up = UpSampling2D(size=2)(pool3) 
        # merge3_2 = Concatenate()([pool2_se, pool3_up])
        # merge3_2 = Conv2D(128, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge3_2)
        # merge3_2 = BatchNormalization(momentum=0.8)(merge3_2)  
        # merge3_2 = resdiual_block(merge3_2, num_filters =128)
        # dec2 = myUpSample2X(dec1, merge3_2, 256) # 128, 128, 384
        merge3_2 = UpSample(pool3, pool2, 128)
        # merge3_2 = squeeze_excite_block(merge3_2)
        dec2 = UpSample(dec1, merge3_2, 256) # 128, 128, 384
        # dec2 = squeeze_excite_block(dec2)
        # print('----------', dec2.shape)

        # add se block ->v1_se
        # pool1_se = squeeze_excite_block(pool1)
        
        # pool2_up = UpSampling2D(size=2)(pool2) 
        # merge2_1 = Concatenate()([pool1_se, pool2_up])
        # merge2_1 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge2_1)
        # merge2_1 = BatchNormalization(momentum=0.8)(merge2_1)  
        # merge2_1 = resdiual_block(merge2_1, num_filters = 64)
        # dec3 = myUpSample2X(dec2, merge2_1, 128) # 256, 256, 192
        merge2_1 = UpSample(pool2, pool1, 64)
        # merge2_1 = squeeze_excite_block(merge2_1)
        dec3 = UpSample(dec2, merge2_1, 128) # 256, 256, 192
        # dec3 = squeeze_excite_block(dec3)
        # print('----------', dec3.shape)


        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        # dec4 = squeeze_excite_block(dec4)
        # dec4 = squeeze_excite_block(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(2, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # print('----------', dec4.shape)
        

        ## return output layer
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(vgg.input, out)

class RUSnet_vgg19():
    """ 
       The SUIM-Net model (Fig. 5 in the paper)
        - base = 'RSB' for RSB-based encoder (Fig. 5b)
        - base = 'VGG' for 12-layer VGG-16 encoder (Fig. 5c)
    """
    

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
        # print(vgg.summary())
        # encoder
        pool1 = vgg.get_layer('block1_pool').output # 256, 256, 64 
        # print('+++++++++++', pool1.shape)
        pool2 = vgg.get_layer('block2_pool').output # 128, 128, 128
        # print('+++++++++++', pool2.shape)
        pool3 = vgg.get_layer('block3_pool').output # 64, 64, 256
        # print('+++++++++++', pool3.shape)
        pool4 = vgg.get_layer('block4_pool').output # 32, 32, 512
        # print('+++++++++++', pool4.shape)
        # print('+++++++++++', pool4.shape)
        ## decoder
        # add se block ->v1_se
        # pool3_se = squeeze_excite_block(pool3)  

        # pool4_up = UpSampling2D(size=2)(pool4) 
        # merge4_3 = Concatenate()([pool3_se, pool4_up])
        # merge4_3 = Conv2D(256, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge4_3)
        # merge4_3 = BatchNormalization(momentum=0.8)(merge4_3)
        # # print('-------------------------', merge4_3.shape)
        # merge4_3 = resdiual_block(merge4_3, num_filters = 256)
        merge4_3 = UpSample(pool4, pool3, 256)
        dec1 = UpSample(pool4, merge4_3, 512) # 64, 64, 768
        # dec1 = squeeze_excite_block(dec1)
        # print('----------', dec1.shape)

        # add se block ->v1_se
        # pool2_se = squeeze_excite_block(pool2)
        
        # pool3_up = UpSampling2D(size=2)(pool3) 
        # merge3_2 = Concatenate()([pool2_se, pool3_up])
        # merge3_2 = Conv2D(128, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge3_2)
        # merge3_2 = BatchNormalization(momentum=0.8)(merge3_2)  
        # merge3_2 = resdiual_block(merge3_2, num_filters =128)
        merge3_2 = UpSample(pool3, pool2, 128)
        dec2 = UpSample(dec1, merge3_2, 256) # 128, 128, 384
        # dec2 = squeeze_excite_block(dec2)
        # print('----------', dec2.shape)

        # add se block ->v1_se
        # pool1_se = squeeze_excite_block(pool1)
        
        # pool2_up = UpSampling2D(size=2)(pool2) 
        # merge2_1 = Concatenate()([pool1_se, pool2_up])
        # merge2_1 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge2_1)
        # merge2_1 = BatchNormalization(momentum=0.8)(merge2_1)  
        # merge2_1 = resdiual_block(merge2_1, num_filters = 64)
        merge2_1 = UpSample(pool2, pool1, 64)
        dec3 = UpSample(dec2, merge2_1, 128) # 256, 256, 192
        # dec3 = squeeze_excite_block(dec3)
        # print('----------', dec3.shape)


        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        # dec4 = squeeze_excite_block(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(2, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # print('----------', dec4.shape)
        

        ## return output layer
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(vgg.input, out)
class RUSnet_resnet50():
    """ 
       The SUIM-Net model (Fig. 5 in the paper)
        - base = 'RSB' for RSB-based encoder (Fig. 5b)
        - base = 'VGG' for 12-layer VGG-16 encoder (Fig. 5c)
    """
    

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
        # print(resnet50.summary())
        # input()
        # encoder
        pool1 = resnet50.get_layer('activation_1').output # 256, 256, 64 
        # print('+++++++++++', pool1.shape)
        pool2 = resnet50.get_layer('activation_10').output # 128, 128, 128
        # print('+++++++++++', pool2.shape)
        pool3 = resnet50.get_layer('activation_22').output # 64, 64, 256
        # print('+++++++++++', pool3.shape)
        pool4 = resnet50.get_layer('activation_40').output # 32, 32, 512
        # print('+++++++++++', pool4.shape)
        # print('+++++++++++', pool4.shape)
        # +++++++++++ (?, 120, 160, 64)
        # +++++++++++ (?, 60, 80, 128)
        # +++++++++++ (?, 30, 40, 256)
        # +++++++++++ (?, 15, 20, 512)

        ## decoder
        # add se block ->v1_se
        # pool3_se = squeeze_excite_block(pool3)  

        # pool4_up = UpSampling2D(size=2)(pool4) 
        # merge4_3 = Concatenate()([pool3_se, pool4_up])
        # merge4_3 = Conv2D(256, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge4_3)
        # merge4_3 = BatchNormalization(momentum=0.8)(merge4_3)
        # # print('-------------------------', merge4_3.shape)
        # merge4_3 = resdiual_block(merge4_3, num_filters = 256)
        merge4_3 = UpSample(pool4, pool3, 256)
        dec1 = UpSample(pool4, merge4_3, 512) # 64, 64, 768
        # dec1 = squeeze_excite_block(dec1)
        # print('----------', dec1.shape)

        # add se block ->v1_se
        # pool2_se = squeeze_excite_block(pool2)
        
        # pool3_up = UpSampling2D(size=2)(pool3) 
        # merge3_2 = Concatenate()([pool2, pool3_up])
        # merge3_2 = Conv2D(128, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge3_2)
        # merge3_2 = BatchNormalization(momentum=0.8)(merge3_2)  
        # merge3_2 = resdiual_block(merge3_2, num_filters =128)
        merge3_2 = UpSample(pool3, pool2, 128)
        dec2 = UpSample(dec1, merge3_2, 256) # 128, 128, 384
        # dec2 = squeeze_excite_block(dec2)
        # print('----------', dec2.shape)

        # add se block ->v1_se
        # pool1_se = squeeze_excite_block(pool1)
        
        # pool2_up = UpSampling2D(size=2)(pool2) 
        # merge2_1 = Concatenate()([pool1, pool2_up])
        # merge2_1 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(merge2_1)
        # merge2_1 = BatchNormalization(momentum=0.8)(merge2_1)  
        # merge2_1 = resdiual_block(merge2_1, num_filters = 64)
        merge2_1 = UpSample(pool2, pool1, 64)
        dec3 = UpSample(dec2, merge2_1, 128) # 256, 256, 192
        # dec3 = squeeze_excite_block(dec3)
        # print('----------', dec3.shape)


        dec4 = UpSampling2D(size=2)(dec3)     # 512, 512, 192
        # dec4 = squeeze_excite_block(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(64, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # dec4 = Conv2D(2, kernel_size = 3, strides = 1, padding='same', activation='relu')(dec4)
        # print('----------', dec4.shape)
        

        ## return output layer
        out = Conv2D(n_classes, (3, 3), padding='same', activation='sigmoid', name='output')(dec4)

        return Model(resnet50.input, out)

if __name__=="__main__":
    net = RUSnet_vgg19(im_res=(512, 512, 3))