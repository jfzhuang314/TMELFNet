import time

from omegaconf import DictConfig

from configs.hydra_yaml_config import instantiate_from_yaml_config
from common.interfaces import BaseLoggerManager, PLCheckpointManager
import pytorch_lightning as pl


class BaseTrainer:
    """Base trainer class that defines common training and prediction interfaces"""
    def __init__(self, config_path="configs/exp", config_name="ELFNet"):

        cfg_obj = instantiate_from_yaml_config(config_path=config_path, config_name=config_name)

        self.cfg_obj = cfg_obj
        self.ds_obj = cfg_obj.dataset
        self.logger_obj = cfg_obj.logger
        self.ckpt_obj = cfg_obj.ckpt
        self.model_obj = cfg_obj.model
        self.trainer = None
        self.module = None
        self.data_module = None
        self.train_loader = None
        self.test_loader = None
        self.loggerMgr = None
        self.ckpt_mgr = None
        self.loaded_ckpt = None

    def build_module(self, model_obj: DictConfig, ckpt):
        """Get model instance - subclasses need to implement this method"""
        raise NotImplementedError("Subclasses must implement the get_module method")

    def prepare_data(self, cfg_obj: DictConfig):
        """Prepare data loaders - subclasses can override this method"""
        raise NotImplementedError("Subclasses must implement the prepare_data method")

    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    def get_module(self):
        """Get model instance"""
        return self.module

    def get_cfg_obj(self):
        """Get configuration object"""
        return self.cfg_obj

    def set_cfg_obj(self, cfg_obj: DictConfig):
        """Set configuration object"""
        self.cfg_obj = cfg_obj
        # Also update its child objects
        self.ds_obj = cfg_obj.dataset
        self.logger_obj = cfg_obj.logger
        self.ckpt_obj = cfg_obj.ckpt
        self.model_obj = cfg_obj.model

    def build_logger_mgr(self, logger_obj: DictConfig):
        """Get logger manager"""
        return BaseLoggerManager.from_config(logger_obj)

    def build_ckpt_mgr(self, ckpt_obj: DictConfig = None):
        """Get checkpoint manager"""
        if ckpt_obj is None:
            ckpt_obj = self.ckpt_obj
        return PLCheckpointManager(
            ckpt_root=ckpt_obj.ckpt_path,
            exp_name=ckpt_obj.exp_name,
            monitor_metric=ckpt_obj.monitor_metric,
            monitor_mode=ckpt_obj.monitor_mode,
            save_top_k=ckpt_obj.save_top_k,
            save_last=ckpt_obj.save_last
        )

    def prepare_trainer(self, ckpt_path=None):
        """Prepare model and trainer
        If ckpt_path is not None and exists locally, load it
        """
        # Prepare data
        self.prepare_data(self.cfg_obj)

        # Prepare logger manager
        self.loggerMgr = self.build_logger_mgr(self.logger_obj)

        # Prepare checkpoint manager
        self.ckpt_mgr = self.build_ckpt_mgr()

        # Load latest checkpoint (resume training)
        # self.module, self.loaded_ckpt = self.ckpt_mgr.load_checkpoint(self.module,ckpt_path)
        # if self.loaded_ckpt:
        #     print(f"✅ Loaded latest checkpoint: {self.loaded_ckpt}")
        self.loaded_ckpt = self.ckpt_mgr.latest_checkpoint_path if ckpt_path is None else ckpt_path

        print(f"✅ Loaded latest checkpoint: {self.loaded_ckpt}") if self.loaded_ckpt else print("No checkpoint found")
            # Get model
        self.module = self.build_module(self.model_obj, self.loaded_ckpt)


        # Create PyTorch Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=self.cfg_obj.training.num_epochs,
            logger=self.loggerMgr.loggers,
            log_every_n_steps=20,
            check_val_every_n_epoch=1,
            precision='16-mixed',
            callbacks=[self.ckpt_mgr.get_ckpt_callback()]
        )

    def fit(self, train_loader=None, val_loader=None):
        """Execute training"""
        if not self.trainer:
            self.prepare_trainer()

        start = time.time()
        self.trainer.fit(
            self.module,
            train_dataloaders=train_loader or self.train_loader,
            val_dataloaders=val_loader or self.test_loader,
            ckpt_path=getattr(self, 'loaded_ckpt', None)
        )
        end = time.time()
        running_time = end - start
        print(f'Time cost: {running_time:.5f}s, {running_time / 3600} h\n')

    def predict(self, predict_loader=None, ckpt_path=None):
        """Execute prediction"""
        if not self.trainer:
            self.prepare_trainer()

        # If checkpoint path is specified, use it for prediction
        if ckpt_path:
            predictions = self.trainer.predict(
                self.module,
                dataloaders=predict_loader,
                ckpt_path=ckpt_path
            )
        else:
            predictions = self.trainer.predict(
                self.module,
                dataloaders=predict_loader
            )

        return predictions

    def validate(self, val_loader=None):
        """Execute validation"""
        if not self.trainer:
            self.prepare_trainer()

        val_loader = val_loader or self.test_loader
        return self.trainer.validate(self.module, dataloaders=val_loader)

    def test(self, test_loader=None):
        """Execute testing"""
        if not self.trainer:
            self.prepare_trainer()

        test_loader = test_loader or self.test_loader
        return self.trainer.test(self.module, dataloaders=test_loader)

