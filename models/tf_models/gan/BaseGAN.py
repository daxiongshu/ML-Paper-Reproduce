import tensorflow as tf
from models.tf_models.BaseModel import BaseModel

class GAN(BaseModel):
    def _build_generator(self):
        raise NotImplementedError()

    def _build_discriminator(self):
        raise NotImplementedError()
