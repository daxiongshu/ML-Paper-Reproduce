from models.tf_models.BaseCnnModel import BaseCnnModel
import tensorflow as tf
import numpy as np


class densenet169(BaseCnnModel):

    def _build(self, inputs, drop_rate=0, resize=True, bottleneck=False):
        net_name = "densenet169"
        showfeamap = self.flags.visualize and 'feamap' in self.flags.visualize

        compression = 0.5
        eps = 1e-5
        nb_filter = 64
        growth_rate = 32
        nb_layers = [6,12,32,32]# For DenseNet-169

        with tf.variable_scope(net_name):
            if self.flags.visualize and 'image' in self.flags.visualize:
                tf.summary.image(name="images", tensor=inputs,
                    max_outputs=6, collections=[tf.GraphKeys.IMAGES])

            if resize:
                with tf.name_scope("Resize"):
                    inputs = tf.image.resize_images(inputs,(224,224))
         
            with tf.name_scope("Preprocess"):
                net = self._preprocess_input(inputs)

            layer_name = "%s/block1"%net_name
            with tf.variable_scope(layer_name.split('/')[-1]):
                padding = tf.constant([[0,0],[3,3],[3,3],[0,0]])
                net = tf.pad(net, padding)

                net = self._conv2D(net, ksize=7, in_channel=net.get_shape().as_list()[-1], out_channel=nb_filter,
                    strides=[1,2,2,1], layer_name='%s/conv'%layer_name, padding='VALID',
                    activation = None, use_bias=False)

                net = self._batch_normalization(net, layer_name='%s/batch_norm'%layer_name, eps=eps) 

                net = self._scale(net, "%s/scale"%layer_name)

                net = self._activate(net, activation="relu")

                padding = tf.constant([[0,0],[1,1],[1,1],[0,0]])
                net = tf.pad(net, padding)

                net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', layer_name = '%s/pool'%layer_name)

            for i in range(len(nb_layers)):
                filters = [growth_rate]*nb_layers[i]
                layer_name = "%s/dense%d"%(net_name,i)
                net = self._dense_block(net,layer_name,filters,drop_rate)
                nb_filter += growth_rate*nb_layers[i]
                nb_filter  = int(nb_filter*compression)
                if i==len(nb_layers)-1:
                    break

                layer_name = "%s/trans%d"%(net_name,i)
                #print(layer_name, int(nb_filter*compression))
                net = self.transition_block(net,layer_name,nb_filter,drop_rate)

            layer_name = "%s/block2"%net_name
            with tf.variable_scope(layer_name.split('/')[-1]):
                net = self._batch_normalization(net, layer_name='%s/batch_norm'%layer_name, eps=eps)

                net = self._scale(net, "%s/scale"%layer_name)

                net = self._activate(net, activation="relu")
                self.bottleneck = net
                if bottleneck:
                    return

                h,w = net.get_shape().as_list()[1:3]
                net = tf.nn.avg_pool(net, ksize=[1,h,w,1], strides=[1,h,w,1], padding='VALID')
                net = tf.contrib.layers.flatten(net) 
            
            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=1000,
                layer_name='%s/fc'%net_name)

            self.logit = net

    def transition_block(self,net,layer_name,nb_filter,drop_rate):
        with tf.variable_scope(layer_name.split('/')[-1]):
            net = self.conv_block1(net,"%s/block"%(layer_name),nb_filter//4,1e-5)
            if self.flags.task and "train" in self.flags.task and drop_rate>0:
                net = tf.nn.dropout(net, 1-drop_rate)

            net = tf.nn.avg_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
        return net


    def _dense_block(self,net,name,filters,drop_rate):
        with tf.variable_scope(name.split('/')[-1]):
            inputs = [net]
            for i in range(len(filters)):
                #print(name,i)
                out = 0
                for c,inx in enumerate(inputs):
                    net = self.conv_block1(inx,"%s/block1_%d_%d"%(name,i,c),filters[i],1e-5)
                    out = out+net
                net = out
                    
                if self.flags.task and "train" in self.flags.task and drop_rate>0:
                    net = tf.nn.dropout(net, 1-drop_rate)

                net = self.conv_block2(net,"%s/block2_%d"%(name,i), filters[i],1e-5)

                if self.flags.task and "train" in self.flags.task and drop_rate>0:
                    net = tf.nn.dropout(net, 1-drop_rate)

                inputs.append(net)
        return tf.concat(inputs,axis=3)

    def conv_block1(self, net, layer_name, nb_filter, eps):
        inter_channel = nb_filter * 4
        with tf.variable_scope(layer_name.split('/')[-1]):
            net = self._batch_normalization(net, layer_name='%s/batch_norm'%layer_name, eps = eps)

            net = self._scale(net, "%s/scale"%layer_name)

            net = self._activate(net, activation="relu")

            net = self._conv2D(net, ksize=1, in_channel=net.get_shape().as_list()[-1], out_channel=inter_channel,
                strides=[1,1,1,1], layer_name='%s/conv'%layer_name, padding='VALID',
                activation = None, use_bias=False)

        return net

    def conv_block2(self, net, layer_name, nb_filter, eps):
        with tf.variable_scope(layer_name.split('/')[-1]):
            net = self._batch_normalization(net, layer_name='%s/batch_norm'%layer_name, eps = eps)

            net = self._scale(net, "%s/scale"%layer_name)

            net = self._activate(net, activation="relu")

            padding = tf.constant([[0,0],[1,1],[1,1],[0,0]])
            net = tf.pad(net, padding)

            net = self._conv2D(net, ksize=3, in_channel=net.get_shape().as_list()[-1], out_channel=nb_filter,
                strides=[1,1,1,1], layer_name='%s/conv'%layer_name, padding='VALID',
                activation = None, use_bias=False)
        return net


    def _preprocess_input(self, x):
        mean = tf.constant([103.94,116.78,123.68])
        x = (x-mean)* 0.017
        return x



