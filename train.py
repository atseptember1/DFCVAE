import tensorflow as tf
import numpy as np 
import os 
import cv2 
import time 

from dfcvae import get_extractor
from utils import progressBar
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications.vgg19 import preprocess_input

        
class Trainer():
    def __init__(self, 
                 progress_dir,
                 checkpoint_dir,
                 encoder, 
                 decoder, 
                 vae_net,
                 z_dim, 
                 test_size,
                 batch_size,
                 learning_rate):
                
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.test_size = test_size
        self.progress_dir = progress_dir
        self.test_points = tf.random.normal(shape=(test_size**2, z_dim))
        
        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                                        optimizer=optimizers.Adam(lr=learning_rate),
                                        encoder=encoder,
                                        decoder=decoder,
                                        vae_net=vae_net)
            
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory=checkpoint_dir,
                                                       max_to_keep=5)
        self.restore_checkpoint()
        
        self.vae_metric = tf.metrics.Mean()
        self.mse_loss = MeanSquaredError()
        
    def restore_checkpoint(self):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Model restored at epoch {self.ckpt.epoch.numpy()}")
            
    def loss_fn(self, image, re_image, mu, log_var):
        pass
            
    @tf.function 
    def train_step(self, train_images):
        with tf.GradientTape() as tape:
            
            re_images, mu, log_var = self.ckpt.vae_net(train_images, training=True)
            loss = self.loss_fn(train_images, re_images, mu, log_var)
        
        grads = tape.gradient(loss, self.ckpt.vae_net.trainable_variables)
        self.ckpt.optimizer.apply_gradients(zip(grads, self.ckpt.vae_net.trainable_variables))
        
        self.vae_metric.update_state(loss)
        return loss
        
    def train_loop(self, dataset, epochs, total_steps):
        for epoch in range(epochs):
            
            start = time.time()
            for i, train_images in enumerate(dataset):
                
                loss = self.train_step(train_images)
                
                progressBar(epoch+1, i+1, total_steps, loss.numpy())
        
                if (i+1) % 50 == 0:
                    self.generate_training_progress_result(self.ckpt.decoder, epoch+1, i+1)
            stop = time.time()
            
            print()
            print(f"EPOCH: {epoch+1} - LOSS: {self.vae_metric.result().numpy()} - Time: {round(stop-start,2)}s")
            
            self.ckpt_manager.save()
            self.ckpt.epoch.assign_add(1)
            self.vae_metric.reset_states()
            
    def generate_training_progress_result(self, model, epoch, step):
        
        test_images = model(self.test_points)
        test_images = test_images.numpy().astype("uint8")
        
        _, height, width, depth = test_images.shape
        
        test_images = test_images.reshape(self.test_size, 
                                          self.test_size, 
                                          height, 
                                          width, 
                                          depth)
        test_images = test_images.transpose(0,2,1,3,4)
        test_images = test_images.reshape(height*self.test_size,
                                          width*self.test_size,
                                          depth)
        path = os.path.join(self.progress_dir, f"Epoch_{epoch}_on_{step}_batch.png")
        cv2.imwrite(path, test_images[...,::-1])
        
class VaeTrainer(Trainer):
    def __init__(self, 
                 progress_dir,
                 checkpoint_dir,
                 encoder, 
                 decoder, 
                 vae_net,
                 reconstruction_weight,
                 z_dim, 
                 test_size,
                 batch_size,
                 learning_rate):
        super(VaeTrainer, self).__init__(progress_dir,
                                         checkpoint_dir,
                                         encoder, 
                                         decoder, 
                                         vae_net,
                                         z_dim, 
                                         test_size,
                                         batch_size,
                                         learning_rate)
        
        self.reconstruction_weight = reconstruction_weight
        
    def loss_fn(self, image, re_image, mu, log_var):
        kl_loss = -0.5*tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        rec_loss = self.mse_loss(image, re_image)
        return kl_loss + self.reconstruction_weight*rec_loss
    
class DfcVaeTrainer(Trainer):
    def __init__(self, 
                 progress_dir,
                 checkpoint_dir,
                 encoder, 
                 decoder, 
                 vae_net,
                 vgg_layers,
                 perceptual_weight,
                 z_dim, 
                 test_size,
                 batch_size,
                 learning_rate):
        super(DfcVaeTrainer, self).__init__(progress_dir,
                                            checkpoint_dir,
                                            encoder, 
                                            decoder, 
                                            vae_net,
                                            z_dim, 
                                            test_size,
                                            batch_size,
                                            learning_rate)
        
        self.extractor = get_extractor(vgg_layers)
        self.perceptual_weight = perceptual_weight
        
    def loss_fn(self, image, re_image, mu, log_var):
        kl_loss = -0.5*tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        p_loss = self.perceptual_loss(image, re_image)
        return kl_loss + self.perceptual_weight*p_loss
    
    @tf.function
    def perceptual_loss(self, ori_img, re_img):    
        ori_img = preprocess_input(ori_img)
        re_img = preprocess_input(re_img)
        ori_features = self.extractor(ori_img)
        re_features = self.extractor(re_img)
        total_loss = 0.0
        for ori, re in zip(ori_features, re_features):
            ori = (3./tf.cast(tf.shape(ori)[-1], tf.float32))*ori 
            re = (3./tf.cast(tf.shape(re)[-1], tf.float32))*re
            total_loss = total_loss + self.mse_loss(ori, re)
        return tf.reduce_sum(total_loss)