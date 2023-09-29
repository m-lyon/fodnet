import lightning.pytorch as pl

from fodnet.core.model import FODNetLightningModel
from fodnet.core.dataset import Subject, FODNetDataModule


def main():
    '''Main function'''
    model = FODNetLightningModel()
    data_module = FODNetDataModule(
        train_subjects=[
            Subject(
                lowres_fod='/home/matt/Dev/debug_hcp_dataset/100206/Diffusion/fod_lowres_10in/fodnet_lowres/wmfod_norm.npy',
                highres_fod='/home/matt/Dev/debug_hcp_dataset/100206/Diffusion/fodnet_highres/wmfod_norm.npy',
                brain_mask='/home/matt/Dev/debug_hcp_dataset/100206/Diffusion/nodif_brain_mask.npy',
            ),
        ],
        val_subjects=[
            Subject(
                lowres_fod='/home/matt/Dev/debug_hcp_dataset/100307/Diffusion/fod_lowres_10in/fodnet_lowres/wmfod_norm.npy',
                highres_fod='/home/matt/Dev/debug_hcp_dataset/100307/Diffusion/fodnet_highres/wmfod_norm.npy',
                brain_mask='/home/matt/Dev/debug_hcp_dataset/100307/Diffusion/nodif_brain_mask.npy',
            ),
        ],
        batch_size=8,
        num_workers=6,
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=1,
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
