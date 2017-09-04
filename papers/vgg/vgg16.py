import tensorflow as tf
from models.tf_models.BaseCnnModel import BaseCnnModel
import time
import numpy as np
import pandas as pd
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]

class vgg16(BaseCnnModel):
    def _build(self, inputs, bottleneck=False):
        # inputs is an image tensor
        # default is NHWC
        # let's build a vgg16
        showfeamap = self.flags.visualize and "feamap" in self.flags.visualize
        assert "vgg16" == self.flags.net
        assert self.flags.color == 3
        net_name = "vgg16"
        with tf.variable_scope(net_name):

            if self.flags.visualize and 'image' in self.flags.visualize:
                tf.summary.image(name="images", tensor=inputs, 
                    max_outputs=6, collections=[tf.GraphKeys.IMAGES])

            with tf.name_scope("Resize"):
                inputs = tf.image.resize_images(inputs,(224,224))

            with tf.name_scope("RGB_to_BGR"):
                bgr = self.rgb_to_bgr(inputs)

            if showfeamap:    
                tf.summary.histogram(name='input', values=bgr, collections=[tf.GraphKeys.FEATURE_MAPS])    
            net = self._conv2D(bgr, ksize=3, in_channel=3, out_channel=64, 
                strides=[1,1,1,1], layer_name='%s/conv1_1'%net_name, padding='SAME',
                activation = 'relu')

            if showfeamap:
                tf.summary.histogram(name='conv1_1', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            net = self._conv2D(net, ksize=3, in_channel=64, out_channel=64,
                strides=[1,1,1,1], layer_name='%s/conv1_2'%net_name, padding='SAME',
                activation = 'relu')

            net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1], 
                padding = 'SAME', layer_name = '%s/pool1'%net_name)

            net = self._conv2D(net, ksize=3, in_channel=64, out_channel=128,
                strides=[1,1,1,1], layer_name='%s/conv2_1'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=128, out_channel=128,
                strides=[1,1,1,1], layer_name='%s/conv2_2'%net_name, padding='SAME',
                activation = 'relu') 

            net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1],
                padding = 'SAME', layer_name = '%s/pool2'%net_name)

            net = self._conv2D(net, ksize=3, in_channel=128, out_channel=256,
                strides=[1,1,1,1], layer_name='%s/conv3_1'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=256, out_channel=256,
                strides=[1,1,1,1], layer_name='%s/conv3_2'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=256, out_channel=256,
                strides=[1,1,1,1], layer_name='%s/conv3_3'%net_name, padding='SAME',
                activation = 'relu')

            net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1],
                padding = 'SAME', layer_name = '%s/pool3'%net_name)

            net = self._conv2D(net, ksize=3, in_channel=256, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv4_1'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=512, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv4_2'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=512, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv4_3'%net_name, padding='SAME',
                activation = 'relu')

            net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1],
                padding = 'SAME', layer_name = '%s/pool4'%net_name) 
           
            self.out4 = net

            net = self._conv2D(net, ksize=3, in_channel=512, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv5_1'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=512, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv5_2'%net_name, padding='SAME',
                activation = 'relu')

            net = self._conv2D(net, ksize=3, in_channel=512, out_channel=512,
                strides=[1,1,1,1], layer_name='%s/conv5_3'%net_name, padding='SAME',
                activation = 'relu')

            net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1],
                padding = 'SAME', layer_name = '%s/pool5'%net_name)
           
            self.bottleneck_conv = net

            with tf.name_scope("flatten"):
                net = tf.contrib.layers.flatten(net)

            self.bottleneck = net

            if showfeamap:
                tf.summary.histogram(name='bottleneck', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])
            
            if bottleneck:
                return

            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=4096, 
                layer_name='%s/fc6'%net_name,
                activation = 'relu')

            if self.flags.task == 'train':
                net = tf.nn.dropout(net, 0.5, name='dropout1')

            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=4096, 
                layer_name='%s/fc7'%net_name,
                activation = 'relu')

            if self.flags.task == 'train':
                net = tf.nn.dropout(net, 0.5, name='dropout2')

            self.fc7_out = net

            if showfeamap:
                tf.summary.histogram(name='fc7_out', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=1000, 
                layer_name='%s/fc8'%net_name)

            self.logit = net

