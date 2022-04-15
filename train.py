import click

from training.training_loop import training_loop

#----------------------------------------------------------------------------

@click.command()

# Required arguments
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--data',         help='Training data', metavar='DIR',                            type=str, required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--epochs',       help='Number of epochs', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--image_size',   help='Image size', metavar='INT',                               type=click.IntRange(min=1), required=True)

# Optional arguments
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), default=1)
@click.option('--workers',      help='Number of workers to use', metavar='INT',                type=click.IntRange(min=1), default=2)
@click.option('n_channels',      help='Number of channels', metavar='INT',                      type=click.IntRange(min=1), default=3)

# Hyperparameters
@click.option('--lr',           help='Learning rate', metavar='FLOAT',                          type=click.FloatRange(min=0.0), default=0.0002)
@click.option('--beta1',        help='Adam beta1', metavar='FLOAT',                             type=click.FloatRange(min=0.0, max=1.0), default=0.5)
@click.option('--z_dim',        help='Latent space dimension', metavar='INT',                    type=click.IntRange(min=1), default=100)
@click.option('--gen_features', help='Number of features in the generator', metavar='INT',       type=click.IntRange(min=1), default=64)
@click.option('--disc_features',help='Number of features in the discriminator', metavar='INT',     type=click.IntRange(min=1), default=64)

def main(**kwargs):
    opt = kwargs
    print(opt)

    training_loop(
        dataroot=opt['data'],
        num_epochs=opt['epochs'],
        workers=opt['workers'],
        batch_size=opt['batch'],
        image_size=opt['image_size'],
        num_channels=opt['n_channels'],
        z_dim=opt['z_dim'],
        gen_features_size=opt['gen_features'],
        disc_features_size=opt['disc_features'],
        lr=opt['lr'],
        beta1=opt['beta1'],
        ngpu=opt['gpus'],
    )

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------