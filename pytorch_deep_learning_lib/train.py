from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

def train_model(data_module, model, max_epochs=200, patience=10):
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=True)
    trainer = Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback], accelerator="gpu", devices="auto")
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    return model
