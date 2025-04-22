from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class StenUNetPretrainedTrainer(nnUNetTrainer):
    def __init__(self, *args,
                 freeze_encoder_until=50,
                 initial_lr=5e-4,
                 weight_decay=1e-4,
                 num_epochs=250,
                 oversample_foreground_percent=0.33,
                 num_iterations_per_epoch=400,
                 num_val_iterations_per_epoch=100,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Custom hyperparameters
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.oversample_foreground_percent = oversample_foreground_percent
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_val_iterations_per_epoch = num_val_iterations_per_epoch

        # Encoder freeze logic
        self.freeze_encoder_until = freeze_encoder_until

    def initialize(self):
        super().initialize()
        self._freeze_encoder()

    def _freeze_encoder(self):
        if hasattr(self.network, 'encoder'):
            for param in self.network.encoder.parameters():
                param.requires_grad = False

    def _unfreeze_encoder(self):
        if hasattr(self.network, 'encoder'):
            for param in self.network.encoder.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.current_epoch == self.freeze_encoder_until:
            self.print_to_log_file(f"\U0001F513 Unfreezing encoder at epoch {self.current_epoch}")
            self._unfreeze_encoder()