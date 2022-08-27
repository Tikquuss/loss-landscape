"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import pytorch_lightning as pl

class Evaluator():
    def __init__(self, metrics : list = None, get_loss = None):
        if get_loss is None :
            trainer_config = {
                "accelerator" : 'auto',
                "devices" : 'auto',
            }
            self.trainer = pl.Trainer(**trainer_config)
            self.metrics = metrics
        else :
            self.get_loss = get_loss

    def __call__(self, pl_module, data_loader):
        results = self.trainer.test(pl_module, data_loader, verbose=False)
        return [results[0][m] for m in self.metrics]

    def _get_loss(self, pl_module, batch):
        """
        Given a batch of data, this function returns the  loss
        """
        return self.get_loss(pl_module, batch)
