import torch
from torch import nn


class GANLoss(nn.Module):
    def __init__(self, conf):
        self.conf = conf
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(self.conf.real))
        self.register_buffer('fake_label', torch.tensor(self.conf.fake))
        gan_mode = self.conf.gan_mode
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.type_as(prediction).expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if not for_discriminator:
                loss = -prediction.mean()
            else:
                if target_is_real:
                    loss = torch.nn.ReLU()(1.0 - prediction).mean()
                else:
                    loss = torch.nn.ReLU()(1.0 + prediction).mean()
        return loss
