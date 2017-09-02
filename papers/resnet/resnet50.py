from models.tf_models.BaseCnnModel import BaseCnnModel
import tensorflow as tf
import numpy as np


class resnet50(BaseCnnModel):

    def _build(self, inputs, bottleneck=False, resize=True, breakpoint = 'out4', activation='relu'):
        net_name = "resnet50"
        showfeamap = self.flags.visualize and 'feamap' in self.flags.visualize
        with tf.variable_scope(net_name):
            if self.flags.visualize and 'image' in self.flags.visualize:
                tf.summary.image(name="images", tensor=inputs,
                    max_outputs=6, collections=[tf.GraphKeys.IMAGES])

            if resize:
                with tf.name_scope("Resize"):
                    inputs = tf.image.resize_images(inputs,(224,224))

            with tf.name_scope("RGB_to_BGR"):
                bgr = self.rgb_to_bgr(inputs)

            self.inputs = bgr
            if showfeamap:
                tf.summary.histogram(name='input', values=self.inputs, collections=[tf.GraphKeys.FEATURE_MAPS])

            net = self._conv2D(self.inputs, ksize=7, in_channel=3, out_channel=64,
                strides=[1,2,2,1], layer_name='%s/conv1'%net_name, padding='SAME',
                activation = None)
            net = self._batch_normalization(net, layer_name='%s/batch_norm1'%net_name)
            net = self._activate(net, activation) 
            
            if showfeamap:
                tf.summary.histogram(name='conv1', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            #print(net.get_shape())

            net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                padding = 'SAME', layer_name = '%s/pool1'%net_name)
            
            net = self._resnet_module(net,'%s/module1'%net_name,blocks=3,ksizes=[1,3,1],filters=[64,64,256], 
                    stride=1, activation = activation)

            if showfeamap:
                tf.summary.histogram(name='module1', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            self.out1 = net
            if breakpoint == "out1":
                return

            net = self._resnet_module(net,'%s/module2'%net_name,blocks=4,ksizes=[1,3,1],filters=[128,128,512], 
                stride=2, activation = activation)
            if showfeamap:
                tf.summary.histogram(name='module2', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            self.out2 = net

            if breakpoint == "out2":
                return

            net = self._resnet_module(net,'%s/module3'%net_name,blocks=6,ksizes=[1,3,1],filters=[256,256,1024],
                stride=2, activation = activation)

            if showfeamap:
                tf.summary.histogram(name='module3', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])

            self.out3 = net
            if breakpoint == "out3":
                return

            net = self._resnet_module(net,'%s/module4'%net_name,blocks=3,ksizes=[1,3,1],filters=[512,512,2048], 
                stride=2, activation = activation)

            self.bottleneck = net
            if bottleneck:
                return

            if showfeamap:
                tf.summary.histogram(name='module4', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])
            #print(net.get_shape())
            h,w = net.get_shape().as_list()[1:3]
            net  = tf.nn.avg_pool(net, ksize=[1,h,w,1], strides=[1,h,w,1], padding='VALID')
            
            #print(net.get_shape())
            with tf.name_scope("flatten"):
                net = tf.contrib.layers.flatten(net)
            
            #print(net.get_shape())
            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=1000, layer_name='%s/fc'%net_name)
            if showfeamap:
                tf.summary.histogram(name='fc', values=net, collections=[tf.GraphKeys.FEATURE_MAPS])
            self.logit = net


    def _resnet_module(self,net,name,blocks,ksizes,filters, stride=1, activation='relu'):
        with tf.variable_scope(name.split('/')[-1]):
            for i in range(1,blocks+1):
                with tf.variable_scope("shortcut%d"%i):
                    lname = '%s/shortcut%d'%(name,i)
                    if i==1:
                        shortcut = self._conv2D(net, ksize=1, in_channel=net.get_shape().as_list()[-1], 
                            out_channel=filters[-1],
                            strides=[1,stride,stride,1], layer_name='%s/conv'%lname, 
                            padding='SAME',
                            activation = None)
                        shortcut = self._batch_normalization(shortcut, layer_name='%s/batch_norm'%lname)
                    else:
                        shortcut = net
                        stride=1

                    
                net = self.conv_block(net, "%s/block%d"%(name,i), ksizes, filters, [activation]*2+[None], [stride,1,1])
                net = net + shortcut

                net = self._activate(net, activation)

        return net
