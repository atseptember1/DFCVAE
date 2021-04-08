import tensorflow as tf

from config import Config as cfg
from tensorflow.keras import layers


########## Encoder ##########
class Sampling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(log_var))
        outputs = mu + tf.exp(0.5*log_var)*epsilon
        return outputs

def conv_block(x, channels, kernel_size):
    x = layers.Conv2D(channels, kernel_size, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x 

def build_encoder(ch, z_dim, image_shape):
    inputs = layers.Input(shape=image_shape)
    x = layers.Lambda(lambda x: x/127.5-1.)(inputs)
    for i in range(4):
        x = conv_block(x, ch, 4)
        ch = ch*2
    x = layers.Flatten()(x)
    mu = layers.Dense(z_dim)(x)
    log_var = layers.Dense(z_dim)(x)
    z = Sampling()((mu, log_var))
    encoder = tf.keras.Model(inputs, [mu, log_var, z])
    return encoder

########## Decoder ##########
class ReplicationPad2D(layers.Layer):
    def __init__(self):
        super(ReplicationPad2D, self).__init__()
        
    def call(self, inputs):
        return tf.pad(inputs, [[0,0], [1,1], [1,1], [0,0]],"SYMMETRIC")
    
def deconv_block(x, channels, kernel_size):
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = ReplicationPad2D()(x)
    x = layers.Conv2D(channels, kernel_size)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_decoder(ch, z_dim):
    inputs = layers.Input(shape=(z_dim))
    x = layers.Dense(4096)(inputs)
    x = layers.Reshape((4,4,256))(x)
    for i in range(3):
        x = deconv_block(x, ch, 3)
        ch = ch // 2 
    x = layers.UpSampling2D(interpolation="nearest")(x)
    x = ReplicationPad2D()(x)
    x = layers.Conv2D(3, 3, activation="tanh")(x)
    outputs = layers.Lambda(lambda x: (x+1.)*127.5)(x)
    decoder = tf.keras.Model(inputs, outputs)
    return decoder

########## VAE ##########
def build_vae(image_shape=cfg.image_shape, 
              z_dim=cfg.z_dim, 
              initial_encoder_channel=32, 
              initial_decoder_channel=128):
    
    encoder = build_encoder(ch=initial_encoder_channel, 
                            z_dim=z_dim, 
                            image_shape=image_shape)
    decoder = build_decoder(ch=initial_decoder_channel, 
                            z_dim=z_dim)

    inputs = layers.Input(shape=image_shape)
    mu, log_var, z = encoder(inputs)
    outputs = decoder(z)
    vae_net = tf.keras.Model(inputs, [outputs, mu, log_var])
    return encoder, decoder, vae_net

######### Extractor #########

def get_extractor(vae_features, image_shape=cfg.image_shape):
    vgg19 = tf.keras.applications.VGG19(include_top=False, input_shape=image_shape)
    vgg19.trainable = False
    
    vgg19_outputs = []
    i = 0
    for layer in vgg19.layers:
        if layer.name == vae_features[i]:
            vgg19_outputs.append(layer.output)
            i = i + 1 
        if i == 3:
            break

    extractor = tf.keras.Model(vgg19.input, vgg19_outputs)
    return extractor 