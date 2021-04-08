import argparse

from config import Config as cfg
from config import VaeConfig, DfcVaeConfig
from utils import validate_path
from data import DataSet
from train import VaeTrainer, DfcVaeTrainer
from dfcvae import build_vae


def main():
    parser = argparse.ArgumentParser(description="Training VAE on CelebA dataset")
    parser.add_argument("--model", type=str, default="VAE", choices=["VAE", "VAE_123", "VAE_345"], help="Training model")
    
    args = vars(parser.parse_args())
    
    datagen = DataSet(cfg.dataset_dir)
    dataset, total_steps = datagen.build(batch_size=cfg.batch_size)
    
    encoder, decoder, vae_net = build_vae(z_dim=cfg.z_dim)
    
    if args["model"] == "VAE":
        
        validate_path(VaeConfig.progress_dir)
        validate_path(VaeConfig.checkpoint_dir)
        
        VAE = VaeTrainer(progress_dir=VaeConfig.progress_dir,
                         checkpoint_dir=VaeConfig.checkpoint_dir,
                         encoder=encoder,
                         decoder=decoder,
                         vae_net=vae_net,
                         reconstruction_weight=VaeConfig.reconstruction_weight,
                         z_dim=cfg.z_dim,
                         test_size=cfg.test_size,
                         batch_size=cfg.test_size,
                         learning_rate=cfg.lr)
    else:
         
        validate_path(DfcVaeConfig.progress_dir)
        validate_path(DfcVaeConfig.checkpoint_dir)
        
        VAE = DfcVaeTrainer(progress_dir=DfcVaeConfig.progress_dir,
                            checkpoint_dir=DfcVaeConfig.checkpoint_dir,
                            encoder=encoder,
                            decoder=decoder,
                            vae_net=vae_net,
                            vgg_layers=DfcVaeConfig.vgg19_layers[args["model"]],
                            perceptual_weight=DfcVaeConfig.perceptual_weight,
                            z_dim=cfg.z_dim,
                            test_size=cfg.test_size,
                            batch_size=cfg.test_size,
                            learning_rate=cfg.lr)
        
    VAE.train_loop(dataset=dataset, 
                   epochs=cfg.epochs, 
                   total_steps=total_steps)
    
    
if __name__ == "__main__":
    main()