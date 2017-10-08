import tensorflow as tf
from models.tf_models.BaseCnnModel import BaseCnnModel

class GAN(BaseCnnModel):
    def _build_generator(self):
        raise NotImplementedError()

    def _build_discriminator(self):
        raise NotImplementedError()

    
