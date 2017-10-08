import tensorflow as tf
from models.tf_models.gan.BaseGAN import GAN
from utils.tf_utils.utils import conv_out_size_same
from PIL import Image
import time
import os
import numpy as np
from utils.image_utils.sp_util import save_images, image_manifold_size, get_image
class DCGAN(GAN):

    def _build(self):
        crop = self.flags.crop
        B,C = self.flags.batch_size,self.flags.color
        oH,oW = self.flags.out_width,self.flags.out_height
        H,W = self.flags.height,self.flags.width
        img_dims = [B,oH,oW,C] if crop else [B,H,W,C]
        z_dim = self.flags.z_dim
        self.real_imgs = tf.placeholder(tf.float32, img_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, shape=(B,z_dim))

        self.G = self._build_generator()
        self.Sample = self._build_generator(reuse=True)
        self.D_logit_real = self._build_discriminator(self.real_imgs,reuse=False) 
        self.D_logit_fake = self._build_discriminator(self.G,reuse=True)
        self.g_step = tf.Variable(0, name='g_step',trainable=False)
        self.d_step = tf.Variable(0, name='d_step',trainable=False)

    def _build_generator(self, reuse=False):
        netname = "Generator"
        s_h,s_w,B = self.flags.out_width,self.flags.out_height,self.flags.batch_size
        gf_dim,z_dim = self.flags.gf_dim,self.flags.z_dim
        B = self.flags.batch_size
        C = self.flags.color
        with tf.variable_scope(netname, reuse=reuse):
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            net = self._fc(self.z, fan_in=z_dim, fan_out=gf_dim*8*s_h16*s_w16, layer_name='%s/fc0'%netname, 
                activation=None) # why *8?

            net = tf.reshape(net, [-1, s_h16, s_w16, gf_dim * 8])
            net = self._batch_normalization(net, layer_name="%s/batch_norm0"%netname, 
                training= (reuse==False)) 
            net = tf.nn.relu(net)

            outshapes = [[B,s_h8, s_w8, gf_dim*4],
                [B,s_h4, s_w4, gf_dim*2],
                [B,s_h2, s_w2, gf_dim],
                [B,s_h, s_w, C]]

            net = self.deconv_block(net, name="%s/deconv_block"%netname, 
                ksizes=[5,5,5,5], outshapes=outshapes,
                activations=['relu']*3+[None], strides=[2,2,2,2],
                batchnorm=[1,1,1,0],
                args={"training":reuse==False,'eps':1e-5,'momentum':0.9})

            return tf.nn.tanh(net)

    def _build_discriminator(self,inputs,reuse):
        netname = "Discriminator"
        df_dim = self.flags.df_dim
        B = self.flags.batch_size
        with tf.variable_scope(netname,reuse=reuse):
            net = self.conv_block(inputs, name="%s/conv_block"%netname, 
                ksizes=[5,5,5,5], filters=[df_dim,df_dim*2,df_dim*4,df_dim*8], 
                activations=['leaky']*4, strides=[2,2,2,2],batchnorm=[0,1,1,1],
                args={'alpha':0.2,'eps':1e-5,'training':True,'momentum':0.9})
            net = tf.reshape(net,[B,-1])
            net = self._fc(net, fan_in=net.get_shape().as_list()[-1], fan_out=1, 
                layer_name='%s/fc0'%netname,activation=None)
            D_logit = net
            return D_logit


    def _get_loss(self):
        with tf.name_scope("Loss"):
            with tf.name_scope("d_loss_real"):
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_real,
                    labels=tf.ones_like(self.D_logit_real)))
            with tf.name_scope("d_loss_fake"):
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_fake,
                    labels=tf.zeros_like(self.D_logit_fake)))
            with tf.name_scope("g_loss"):
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logit_fake,
                    labels=tf.ones_like(self.D_logit_fake)))
            with tf.name_scope("d_loss"):
                self.d_loss = self.d_loss_real + self.d_loss_fake

        if self.flags.visualize:
            with tf.name_scope("d_loss"):
                tf.summary.scalar(name='d_loss_real', tensor=self.d_loss_real,
                    collections=[tf.GraphKeys.SCALARS])
                tf.summary.scalar(name='d_loss_fake', tensor=self.d_loss_fake,
                    collections=[tf.GraphKeys.SCALARS])
                tf.summary.scalar(name='d_loss', tensor=self.d_loss,
                    collections=[tf.GraphKeys.SCALARS])

            with tf.name_scope("g_loss"):
                tf.summary.scalar(name='g_loss', tensor=self.g_loss,
                    collections=[tf.GraphKeys.SCALARS])
    
    def _get_opt(self):
        # build the self.opt_op for training
        self.set_train_var()
        g_vars = [var for var in self.var_list if var.name.startswith("Generator")]
        d_vars = [var for var in self.var_list if var.name.startswith("Discriminator")]
        #print(len(self.var_list),len(g_vars),len(d_vars))
        self.print_trainable()
        def _select_opt(flags,scaler=1.0):
            lr = self.flags.learning_rate * scaler
            if flags.opt == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5)
            elif flags.opt == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
            elif flags.opt == 'momentum':
                opt = tf.train.MomentumOptimizer(learning_rate=lr,
                    momentum = self.flags.momentum)
            else:
                print("unkown opt %s"%self.flags.opt)
                assert 0
            return opt

        with tf.name_scope("Optimizer"):
            d_opt = _select_opt(self.flags,scaler=1.0)
            g_opt = _select_opt(self.flags,scaler=1.0)
            #self.g_opt_op = g_opt.minimize(self.g_loss, var_list=g_vars)
            #self.d_opt_op = d_opt.minimize(self.d_loss, var_list=d_vars)

            d_grads = tf.gradients(self.d_loss, d_vars)
            d_grads_vars = list(zip(d_grads, d_vars))
            self.d_opt_op = d_opt.apply_gradients(grads_and_vars=d_grads_vars,
                global_step = self.d_step)

            g_grads = tf.gradients(self.g_loss, g_vars)

            g_grads_vars = list(zip(g_grads, g_vars))
            self.g_opt_op = g_opt.apply_gradients(grads_and_vars=g_grads_vars,
                global_step = self.g_step)

            if self.flags.visualize and "grad" in self.flags.visualize:
                for grad, var in g_grads_vars:
                    tf.summary.histogram(var.name + '/gradient', 
                        grad, collections=[tf.GraphKeys.GRADIENTS])
                for grad, var in d_grads_vars:
                    tf.summary.histogram(var.name + '/gradient',
                        grad, collections=[tf.GraphKeys.GRADIENTS])

    def train(self):

        self._build()
        self._get_loss()
        self._get_opt()
        g_sum_op = self._get_summary(tag='g_')
        d_sum_op = self._get_summary(tag='d_')
        d_num,g_num = self.flags.d_num_update,self.flags.g_num_update
        data_path = self.flags.data_path
        start_time = time.time()
        B = self.flags.batch_size
        sample_z = np.random.uniform(-1, 1, [B, self.flags.z_dim]) \
            .astype(np.float32)
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self._restore()
            if self.flags.log_path and self.flags.visualize is not None:
                summary_writer = tf.summary.FileWriter(self.flags.log_path, sess.graph)
            count = 0
            ave_d_loss, ave_g_loss = 0,0
            self.epoch = 0
            for batch in self._batch_gen():
                real_imgs,batch_z,epoch = batch
                for _ in range(d_num):
                    if self.flags.log_path and self.flags.visualize is not None:
                        d_sum,_,d_step,d_loss = sess.run([d_sum_op,self.d_opt_op,self.d_step,self.d_loss],
                            feed_dict={self.real_imgs:real_imgs,self.z:batch_z})
                        summary_writer.add_summary(d_sum, d_step)
                    else:
                        _,d_loss = sess.run([self.d_opt_op,self.d_loss],
                            feed_dict={self.real_imgs:real_imgs,self.z:batch_z})
                ave_d_loss = ave_d_loss*0.99+d_loss*0.01 if count>0 else d_loss
                for _ in range(g_num):
                    if self.flags.log_path and self.flags.visualize is not None:
                        g_sum,_,g_step,g_loss = sess.run([g_sum_op,self.g_opt_op,self.g_step,self.g_loss],
                            feed_dict={self.z:batch_z})
                        summary_writer.add_summary(g_sum, g_step)
                    else:
                        _,g_loss = sess.run([self.g_opt_op,self.g_loss],
                            feed_dict={self.z:batch_z})
                ave_g_loss = ave_g_loss*0.99+g_loss*0.01 if count>0 else g_loss
                count += 1
                if count == 1:
                    print("First d_loss {} g_loss {}".format(ave_d_loss,ave_g_loss))
                if count%self.flags.verbosity==0:
                    duration = time.time() - start_time
                    print("Epochs: %d batches: %d D Loss: %.4f G Loss: %.4f Time: %.3f s"%(
                        self.epoch, count,
                        ave_d_loss, ave_g_loss, duration))
                if count%100 == 0:
                    samples = sess.run(self.Sample,feed_dict={self.z:sample_z})
                    save_images(samples, image_manifold_size(samples.shape[0]),
                        '{}/train_{:02d}_{:04d}.png'.format(data_path, self.flags.pre_epochs+epoch, count))
                if epoch>self.epoch:
                    self.epoch = epoch
                    self._save()
            self.epoch = self.flags.epochs
            self._save()

    def _batch_gen(self):
        path = self.flags.input_path
        B,W,H = self.flags.batch_size,self.flags.width,self.flags.height
        oW,oH = self.flags.out_width,self.flags.out_height
        imgs = os.listdir(path)
        imgs = ["%s/%s"%(path,i) for i in imgs]
        #imgs = [np.array(Image.open(i).resize((W,H))) for i in imgs]
        epochs = self.flags.epochs
        b_per_e = len(range(0,len(imgs),B))
        print("Batches per epoch",b_per_e)
        from random import shuffle
        for epoch in range(epochs):
            shuffle(imgs)
            for i in range(0,len(imgs)-B,B):
                batch_z = np.random.uniform(-1, 1, [B, self.flags.z_dim]) \
                    .astype(np.float32)
                data = [get_image(i,
                    input_height=H,
                    input_width=W,
                    resize_height=oH,
                    resize_width=oW,
                    crop=self.flags.crop,
                    grayscale=0) for i in imgs[i:i+B]]
                #data = [np.array(Image.open(i).resize((W,H))) for i in imgs[i:i+B]]
                yield np.array(data).astype(np.float32), batch_z, epoch
                

    def _get_summary(self, tag = "d_"):
        # build the self.summ_op for tensorboard
        # This function could be overwritten
        if not self.flags.visualize or self.flags.visualize=='none':
            return
        summ_collection = "{} {} {} summaries".format(self.flags.paper, tag, self.flags.run_name)

        for i in tf.get_collection(tf.GraphKeys.SCALARS):
            if tag in i.name:
                tf.add_to_collection(summ_collection, i)
        for i in tf.get_collection(tf.GraphKeys.WEIGHTS):
            if tag in i.name:
                tf.add_to_collection(summ_collection, i)
        for i in tf.get_collection(tf.GraphKeys.FEATURE_MAPS):
            if tag in i.name:
                tf.add_to_collection(summ_collection, i)
        for i in tf.get_collection(tf.GraphKeys.IMAGES):
            if tag in i.name:
                tf.add_to_collection(summ_collection, i)
        for i in tf.get_collection(tf.GraphKeys.GRADIENTS):
            if tag in i.name:
                tf.add_to_collection(summ_collection, i)
        summ_op = tf.summary.merge(tf.get_collection(summ_collection))
        return summ_op

    def _batch_normalization(self, x, layer_name, eps=0.001, training=False, momentum=0.99):
        with tf.variable_scope(layer_name.split('/')[-1]):
            beta, gamma, mean, variance = self._get_batch_normalization_weights(layer_name,
                name = "BatchNorm")
            if beta is None:
                net = tf.contrib.layers.batch_norm(x,
                    decay=momentum,
                    updates_collections=None,
                    epsilon=eps,
                    scale=True,
                    is_training=training)
            else:
                net = tf.contrib.layers.batch_norm(x,
                    decay=momentum,
                    updates_collections=None,
                    epsilon=eps,
                    scale=True,
                    is_training=training,
                    param_initializers={"beta":tf.constant_initializer(value=beta,dtype=tf.float32),
                        'gamma':tf.constant_initializer(value=gamma,dtype=tf.float32),
                        'moving_mean':tf.constant_initializer(value=mean,dtype=tf.float32),
                        'moving_variance':tf.constant_initializer(value=variance,dtype=tf.float32)})
            mean = '%s/BatchNorm/moving_mean:0'%(layer_name)
            variance = '%s/BatchNorm/moving_variance:0'%(layer_name)
            tf.add_to_collection(tf.GraphKeys.SAVE_TENSORS, tf.get_default_graph().get_tensor_by_name(mean))
            tf.add_to_collection(tf.GraphKeys.SAVE_TENSORS, tf.get_default_graph().get_tensor_by_name(variance))
        return net
