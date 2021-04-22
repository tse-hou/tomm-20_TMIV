import tensorflow as tf
import tf_utils

def agent(image, n_actions,train_phase):
    print('image',image)
    with tf.variable_scope('agent'):
        c1_1 = tf_utils.conv_blk(image, 128, [8,8], train_phase, strides=(2,2), name ="c1_1")
        print('c1_1',c1_1)
        c1_2 = tf_utils.conv_blk(c1_1, 128, [3,3], train_phase, name ="c1_2")
        print('c1_2',c1_2)
        #--
        c2_1 = tf_utils.conv_blk(c1_2, 256, [8,8], train_phase, strides=(2,2), name ="c2_1")
        print('c2_1',c2_1)
        c2_2 = tf_utils.conv_blk(c2_1, 256, [3,3], train_phase, name ="c2_2")
        print('c2_2',c2_2)
        #--
        c3_1 = tf_utils.conv_blk(c2_2, 512, [4,4], train_phase, strides=(2,2), name ="c3_1")
        print('c3_1',c3_1)
        c3_2 = tf_utils.conv_blk(c3_1, 512, [3,3], train_phase, name ="c3_2")
        print('c3_2',c3_2)
        #--
        c4_1 = tf_utils.conv_blk(c3_2, 512, [4,4], train_phase, strides=(2,2), name ="c4_1")
        print('c4_1',c4_1)
        c4_2 = tf_utils.conv_blk(c4_1, 512, [3,3], train_phase, name ="c4_2")
        print('c4_2',c4_2)
        #--
        c5_1 = tf_utils.conv_blk(c4_2, 512, [4,4], train_phase, strides=(2,2), name ="c5_1")
        print('c5_1',c5_1)
        c5_2 = tf_utils.conv_blk(c5_1, 512, [3,3], train_phase, name ="c5_2")
        print('c5_2',c5_2)
        flt = tf.layers.flatten(c5_2)
        print('flt',flt)
        f1 = tf_utils.fc_blk(flt, 512, train_phase, name = 'f1')
        print('f1',f1)
        f2 = tf_utils.fc_blk(f1, 512, train_phase, name = 'f2')
        print('f2',f2)
        f3 = tf_utils.fc_blk(f2, 512, train_phase, name = 'f3')
        print('f3',f3)
        out = tf.layers.dense(inputs=f3, units=n_actions, activation=None, name="l3")
        print('out',out)
        return out
    
def gen_agl_map(inputs, height, width, feature_dims):
    with tf.name_scope("gen_agl_map"):
        batch_size = tf.shape(inputs)[0]
        ret = tf.reshape(tf.tile(inputs,tf.constant([1,height*width])), [batch_size,height,width,feature_dims])
        return ret
    
def state_encoder(inputs, height, width, tar_dim, train_phase):
    with tf.variable_scope('state_encoder'):
        dnn_blk_0 = tf_utils.fc_blk(inputs, 32, train_phase, name='dnn_blk_0')
        dnn_blk_1 = tf_utils.fc_blk(dnn_blk_0, 16, train_phase, name='dnn_blk_1')
        dnn_blk_2 = tf_utils.fc_blk(dnn_blk_1, tar_dim, train_phase, name='dnn_blk_2')
        agl_map = gen_agl_map(dnn_blk_2, height, width, tar_dim)
        return agl_map

def model(s, I, D, P, O, train_phase, n_actions):
    with tf.variable_scope('mdl'):
       # image and depth
        I_gray = tf.reduce_mean(I, axis = 4, keep_dims = True)
        D_gray = tf.reduce_mean(D, axis = 4, keep_dims = True)

        imgs_shape = I_gray.get_shape()#(,7,256,256,3)
        I_t = tf.transpose(I_gray, perm=[0,2,3,4,1])# (,256,256,3,7)
        D_t = tf.transpose(D_gray, perm=[0,2,3,4,1])# (,256,256,3,7)
        I_t_reshape = tf.reshape(I_t, [-1, imgs_shape[2].value, imgs_shape[3].value, imgs_shape[1].value*imgs_shape[4].value])
        D_t_reshape = tf.reshape(D_t, [-1, imgs_shape[2].value, imgs_shape[3].value, imgs_shape[1].value*imgs_shape[4].value])
        # position and orientation
        POs = tf.concat([P,O,s], axis = 1)
#         PO_map = state_encoder(PO, imgs_shape[2].value,imgs_shape[3].value,16, train_phase)            
        # state
        pos_rep = gen_agl_map(POs, imgs_shape[2].value,imgs_shape[3].value, 21+21+3)            
        observ = tf.concat([I_t_reshape,D_t_reshape,pos_rep], axis = 3)           
        action = agent(observ,n_actions,train_phase)
        return action