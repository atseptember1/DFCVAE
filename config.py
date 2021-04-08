class Config(object):
    
    image_shape = (64, 64, 3)
    
    dataset_dir = "./data/img_align_celeba"
    checkpoint_dir = None
    progress_dir = None #path to save training progress images
    
    test_size = 8 #Square root number of training progress images
    
    z_dim = 100
    batch_size = 64
    epochs = 5
    lr = 0.0005
    
    
class VaeConfig(Config):
    
    reconstruction_weight = 1e-3
    
    progress_dir = "./data/plain_vae"
    checkpoint_dir = "./checkpoints/plain_vae"
    
class DfcVaeConfig(Config):
    
    perceptual_weight = 0.5
    vgg19_layers = {
        "VAE_123": ["block1_conv1", "block2_conv1", "block3_conv1"],
        "VAE_345": ["block3_conv1", "block4_conv1", "block5_conv1"]
        }