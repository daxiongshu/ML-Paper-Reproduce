from models.tf_models.BaseCnnModel import BaseCnnModel
import tensorflow as tf
import numpy as np


class inception_v3(BaseCnnModel):

    def _build(self, inputs, resize=True):
        net_name = "inception_v3"
        showfeamap = self.flags.visualize and 'feamap' in self.flags.visualize
        with tf.variable_scope(net_name):
            if self.flags.visualize and 'image' in self.flags.visualize:
                tf.summary.image(name="images", tensor=inputs,
                    max_outputs=6, collections=[tf.GraphKeys.IMAGES])

            if resize:
                with tf.name_scope("Resize"):
                    inputs = tf.image.resize_images(inputs,(299,299))

            with tf.name_scope("Preprocess"):
                net = self._preprocess_input(inputs)

            net = self.conv_block(net,"%s/block1"%(net_name), ksizes=[3,3,3], filters=[32,32,64], 
                activations=['relu']*3, strides=[2,1,1], padding=["VALID","VALID","SAME"])

            net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                padding = 'VALID', layer_name = '%s/pool1'%net_name)

            net = self.conv_block(net,"%s/block2"%(net_name), ksizes=[1,3], filters=[80,192],
                activations=['relu']*2, strides=[1,1], padding=["VALID","VALID"])

            net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                padding = 'VALID', layer_name = '%s/pool2'%net_name)

            layer_name = "%s/inception0"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[64],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net_5x5 = self.conv_block(net,"%s/5x5"%(layer_name), ksizes=[1,5], filters=[48,64],
                    activations=['relu']*2, strides=[1]*2, padding=["SAME"]*2)

                net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[1,3,3], filters=[64,96,96],
                    activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[32],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net = tf.concat([net_1x1, net_5x5, net_3x3, net],axis=3)
         

            layer_name = "%s/inception1"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[64],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net_5x5 = self.conv_block(net,"%s/5x5"%(layer_name), ksizes=[1,5], filters=[48,64],
                    activations=['relu']*2, strides=[1]*2, padding=["SAME"]*2)

                net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[1,3,3], filters=[64,96,96],
                    activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[64],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net = tf.concat([net_1x1, net_5x5, net_3x3, net],axis=3)

            layer_name = "%s/inception2"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[64],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net_5x5 = self.conv_block(net,"%s/5x5"%(layer_name), ksizes=[1,5], filters=[48,64],
                    activations=['relu']*2, strides=[1]*2, padding=["SAME"]*2)

                net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[1,3,3], filters=[64,96,96],
                    activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[64],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net = tf.concat([net_1x1, net_5x5, net_3x3, net],axis=3)

            layer_name = "%s/inception3"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[3], filters=[384],
                    activations=['relu'], strides=[2], padding=["VALID"])

                net_3x3d = self.conv_block(net,"%s/3x3d"%(layer_name), ksizes=[1,3,3], filters=[64,96,96],
                    activations=['relu']*3, strides=[1,1,2], padding=["SAME"]*2+['VALID'])

                net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', layer_name = '%s/pool'%net_name)

                net = tf.concat([net_3x3, net_3x3d, net],axis=3)

            layer_name = "%s/inception4"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[192],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net_7x7 = self.conv_block(net,"%s/7x7"%(layer_name), ksizes=[1,[1,7],[7,1]], filters=[128,128,192],
                    activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                net_7x7_db = self.conv_block(net,"%s/7x7_db"%(layer_name), ksizes=[1,[7,1],[1,7],[7,1],[1,7]], filters=[128]*4+[192],
                    activations=['relu']*5, strides=[1]*5, padding=["SAME"]*5)

                net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[192],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net = tf.concat([net_1x1, net_7x7, net_7x7_db, net],axis=3)

            for i in [5,6]:
                layer_name = "%s/inception%d"%(net_name,i)
                with tf.variable_scope(layer_name.split('/')[-1]):
                    net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[192],
                        activations=['relu'], strides=[1], padding=["SAME"])

                    net_7x7 = self.conv_block(net,"%s/7x7"%(layer_name), ksizes=[1,[1,7],[7,1]], filters=[160,160,192],
                        activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                    net_7x7_db = self.conv_block(net,"%s/7x7_db"%(layer_name), ksizes=[1,[7,1],[1,7],[7,1],[1,7]], filters=[160]*4+[192],
                        activations=['relu']*5, strides=[1]*5, padding=["SAME"]*5)

                    net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                    net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[192],
                        activations=['relu'], strides=[1], padding=["SAME"])

                    net = tf.concat([net_1x1, net_7x7, net_7x7_db, net],axis=3)

            layer_name = "%s/inception7"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[192],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net_7x7 = self.conv_block(net,"%s/7x7"%(layer_name), ksizes=[1,[1,7],[7,1]], filters=[192]*3,
                    activations=['relu']*3, strides=[1]*3, padding=["SAME"]*3)

                net_7x7_db = self.conv_block(net,"%s/7x7_db"%(layer_name), ksizes=[1,[7,1],[1,7],[7,1],[1,7]]
, filters=[192]*5,
                    activations=['relu']*5, strides=[1]*5, padding=["SAME"]*5)

                net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[192],
                    activations=['relu'], strides=[1], padding=["SAME"])

                net = tf.concat([net_1x1, net_7x7, net_7x7_db, net],axis=3)

            layer_name = "%s/inception8"%(net_name)
            with tf.variable_scope(layer_name.split('/')[-1]):
                net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[1,3], filters=[192,320],
                    activations=['relu']*2, strides=[1,2], padding=["SAME","VALID"])

                net_7x7 = self.conv_block(net,"%s/7x7"%(layer_name), ksizes=[1,[1,7],[7,1],3], filters=[192]*4,
                    activations=['relu']*4, strides=[1]*3+[2], padding=["SAME"]*3+['VALID'])

                net = self._max_pool2D(net, ksize = [1,3,3,1], strides = [1,2,2,1],
                    padding = 'VALID', layer_name = '%s/pool'%net_name)

                net = tf.concat([net_3x3, net_7x7, net],axis=3)

            for i in [9,10]:
                layer_name = "%s/inception%d"%(net_name,i)
                with tf.variable_scope(layer_name.split('/')[-1]):
                    net_1x1 = self.conv_block(net,"%s/1x1"%(layer_name), ksizes=[1], filters=[320],
                        activations=['relu'], strides=[1], padding=["SAME"])

                    net_3x3 = self.conv_block(net,"%s/3x3"%(layer_name), ksizes=[1], filters=[384],
                        activations=['relu'], strides=[1], padding=["SAME"])
                    net_3x3_1 = self.conv_block(net_3x3,"%s/3x3_1"%(layer_name), ksizes=[[1,3]], filters=[384],
                        activations=['relu'], strides=[1], padding=["SAME"]) 
                    net_3x3_2 = self.conv_block(net_3x3,"%s/3x3_2"%(layer_name), ksizes=[[3,1]], filters=[384],
                        activations=['relu'], strides=[1], padding=["SAME"])
                    net_3x3 = tf.concat([net_3x3_1,net_3x3_2],axis=3)


                    net_3x3_db = self.conv_block(net,"%s/3x3_db"%(layer_name), ksizes=[1,3], filters=[448,384],
                        activations=['relu']*2, strides=[1]*2, padding=["SAME"]*2)

                    net_3x3_db1 = self.conv_block(net_3x3_db,"%s/3x3_db1"%(layer_name), ksizes=[[1,3]], filters=[384],
                        activations=['relu'], strides=[1], padding=["SAME"])
                    net_3x3_db2 = self.conv_block(net_3x3_db,"%s/3x3_db2"%(layer_name), ksizes=[[3,1]], filters=[384],
                        activations=['relu'], strides=[1], padding=["SAME"])
                    net_3x3_db = tf.concat([net_3x3_db1,net_3x3_db2],axis=3)

                    net = tf.nn.avg_pool(net, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                    net = self.conv_block(net,"%s/avg_1x1"%(layer_name), ksizes=[1], filters=[192],
                        activations=['relu'], strides=[1], padding=["SAME"])

                    net = tf.concat([net_1x1, net_3x3, net_3x3_db, net],axis=3) 

            #print(net.get_shape().as_list())
            h,w = net.get_shape().as_list()[1:3]
            net  = tf.nn.avg_pool(net, ksize=[1,h,w,1], strides=[1,h,w,1], padding='VALID')
            with tf.name_scope("flatten"):
                net = tf.contrib.layers.flatten(net)
            #print(net.get_shape().as_list())

            net = self._fc(net, fan_in = net.get_shape().as_list()[1], fan_out=1000, layer_name='%s/fc'%net_name)
            self.logit = net

    def _preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x
