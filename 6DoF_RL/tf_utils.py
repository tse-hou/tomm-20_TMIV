import tensorflow as tf

def batch_norm(x, train_phase, name='bn_layer'):
    #with tf.variable_scope(name) as scope:
    batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training = train_phase,
            name=name
    )
    return batch_norm
    
def conv_blk(inputs, filters, kernel_size, phase_train, strides=(1,1),name = 'conv_blk'):
    with tf.variable_scope(name) as scope:
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding="same",strides=strides, activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="conv")
        #bn = batch_norm(conv, phase_train)
        act = tf.nn.relu(conv, name= "act")
        return act

def fc_blk(inputs, nodes, phase_train, name = 'fc_blk'):
    with tf.variable_scope(name) as scope:
        fc = tf.layers.dense(inputs=inputs, units=nodes, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="fc")
        #bn = batch_norm(fc, phase_train)
        act = tf.nn.relu(fc, name= "act")
        return act
    
def conv_res_blk(inputs, filters, kernel_size, phase_train, name = 'conv_res_blk'):
    with tf.variable_scope(name) as scope:
        conv1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding="same", activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="conv1")
        bn1 = batch_norm(conv1, phase_train, name='bn1')
        act1 = tf.nn.relu(bn1, name= "act1")
        conv2 = tf.layers.conv2d(inputs=act1, filters=filters, kernel_size=kernel_size, padding="same", activation=None, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="conv2")
        bn2 = batch_norm(conv2, phase_train, name='bn2')
        add = inputs + bn2
        act2 = tf.nn.relu(add, name= "act2")
        return act2

def fc_res_blk(inputs, nodes, phase_train, name = 'fc_res_blk'):
    with tf.variable_scope(name) as scope:
        fc1 = tf.layers.dense(inputs=inputs, units=nodes, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="fc1")
        bn1 = batch_norm(fc1, phase_train, name='bn1')
        act1 = tf.nn.relu(bn1, name= "act1")
        fc2 = tf.layers.dense(inputs=act1, units=nodes, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001), name="fc2")
        bn2 = batch_norm(fc2, phase_train, name='bn2')
        add = inputs + bn2
        act2 = tf.nn.relu(add, name= "act2")
        return act2
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")