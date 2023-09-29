'''FOD-Net Model'''

import torch  # pylint: disable=import-error
import lightning.pytorch as pl


class FODNet(torch.nn.Module):
    '''FODNet model'''

    def __init__(self):
        super().__init__()
        self.conv3d1 = torch.nn.Conv3d(in_channels=45, out_channels=256, kernel_size=3)
        self.bn3d1 = torch.nn.BatchNorm3d(256)
        self.glu3d1 = torch.nn.GLU(dim=1)

        self.conv3d2 = torch.nn.Conv3d(in_channels=128, out_channels=512, kernel_size=3)
        self.bn3d2 = torch.nn.BatchNorm3d(512)
        self.glu3d2 = torch.nn.GLU(dim=1)

        self.conv3d3 = torch.nn.Conv3d(in_channels=256, out_channels=1024, kernel_size=3)
        self.bn3d3 = torch.nn.BatchNorm3d(1024)
        self.glu3d3 = torch.nn.GLU(dim=1)

        self.conv3d4 = torch.nn.Conv3d(in_channels=512, out_channels=2048, kernel_size=3)
        self.bn3d4 = torch.nn.BatchNorm3d(2048)
        self.glu3d4 = torch.nn.GLU(dim=1)

        self.joint_linear = torch.nn.Linear(in_features=1024, out_features=2048)
        self.joint_bn = torch.nn.BatchNorm1d(2048)
        self.joint_glu = torch.nn.GLU(dim=1)

        self.l0_pred = CEBlock(num_coeff=2)
        self.l2_pred = CEBlock(num_coeff=10)
        self.l4_pred = CEBlock(num_coeff=18)
        self.l6_pred = CEBlock(num_coeff=26)
        self.l8_pred = CEBlock(num_coeff=34)
        self.apply(init_weights)

    def forward(self, fodlr):
        '''forward pass'''
        x = self.conv3d1(fodlr)
        x = self.bn3d1(x)
        x = self.glu3d1(x)

        x = self.conv3d2(x)
        x = self.bn3d2(x)
        x = self.glu3d2(x)

        x = self.conv3d3(x)
        x = self.bn3d3(x)
        x = self.glu3d3(x)

        x = self.conv3d4(x)
        x = self.bn3d4(x)
        x = self.glu3d4(x)

        x = x.squeeze(2).squeeze(2).squeeze(2)
        x = self.joint_linear(x)
        x = self.joint_bn(x)
        joint = self.joint_glu(x)

        x = self.l0_pred(joint)
        l0_residual = x[:, :1]
        l0_scale = torch.nn.functional.sigmoid(x[:, 1:])

        x = self.l2_pred(joint)
        l2_residual = x[:, :5]
        l2_scale = torch.nn.functional.sigmoid(x[:, 5:])

        x = self.l4_pred(joint)
        l4_residual = x[:, :9]
        l4_scale = torch.nn.functional.sigmoid(x[:, 9:])

        x = self.l6_pred(joint)
        l6_residual = x[:, :13]
        l6_scale = torch.nn.functional.sigmoid(x[:, 13:])

        x = self.l8_pred(joint)
        l8_residual = x[:, :17]
        l8_scale = torch.nn.functional.sigmoid(x[:, 17:])

        residual = torch.cat(
            [l0_residual, l2_residual, l4_residual, l6_residual, l8_residual], dim=1
        )
        scale = torch.cat([l0_scale, l2_scale, l4_scale, l6_scale, l8_scale], dim=1)

        fodpred = residual * scale + fodlr[:, :, 4, 4, 4]

        return fodpred


class CEBlock(torch.nn.Module):
    '''CE Block'''

    def __init__(self, num_coeff):
        super().__init__()
        self.l_0 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.bn_0 = torch.nn.BatchNorm1d(1024)
        self.glu_0 = torch.nn.GLU(dim=1)
        self.l_1 = torch.nn.Linear(in_features=512, out_features=512)
        self.bn_1 = torch.nn.BatchNorm1d(512)
        self.glu_1 = torch.nn.GLU(dim=1)
        self.l_2 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_2 = torch.nn.BatchNorm1d(512)
        self.glu_2 = torch.nn.GLU(dim=1)
        self.l_3 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_3 = torch.nn.BatchNorm1d(512)
        self.glu_3 = torch.nn.GLU(dim=1)
        self.l_4 = torch.nn.Linear(in_features=256, out_features=512)
        self.bn_4 = torch.nn.BatchNorm1d(512)
        self.glu_4 = torch.nn.GLU(dim=1)
        self.pred = torch.nn.Linear(in_features=256, out_features=num_coeff)

    def forward(self, x):
        '''forward pass'''
        x = self.l_0(x)
        x = self.bn_0(x)
        x = self.glu_0(x)
        x = self.l_1(x)
        x = self.bn_1(x)
        x = self.glu_1(x)
        x = self.l_2(x)
        x = self.bn_2(x)
        x = self.glu_2(x)
        x = self.l_3(x)
        x = self.bn_3(x)
        x = self.glu_3(x)
        x = self.l_4(x)
        x = self.bn_4(x)
        x = self.glu_4(x)
        x = self.pred(x)
        return x


def init_weights(module):
    '''Initialize Network Weights.'''

    if isinstance(module, (torch.nn.Conv3d, torch.nn.Linear)):
        gain = torch.nn.init.calculate_gain('leaky_relu')
        torch.nn.init.xavier_uniform_(module.weight.data, gain=gain)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)
    if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm1d)):
        torch.nn.init.normal_(module.weight.data, 1.0, 1.0)
        torch.nn.init.constant_(module.bias.data, 0.0)


class FODNetLightningModel(pl.LightningModule):
    '''FODNet Lightning Model'''

    def __init__(self):
        super().__init__()
        self.fodnet = FODNet()

    @property
    def loss_func(self):
        '''Loss function'''
        return torch.nn.functional.l1_loss

    def configure_optimizers(self):
        '''Optimizer'''
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, train_batch, *args):
        # pylint: disable=arguments-differ
        fod_in, fod_out = train_batch
        fod_out_inf = self.fodnet(fod_in)
        loss = self.loss_func(fod_out, fod_out_inf)
        self.log('train_loss', loss)
        self.log('epoch_train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, *args):
        # pylint: disable=arguments-differ
        fod_in, fod_out = val_batch
        fod_out_inf = self.fodnet(fod_in)

        loss = self.loss_func(fod_out, fod_out_inf)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, *args):
        # pylint: disable=unused-argument
        return self.fodnet(batch)
