#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    graph     = tf.get_default_graph()
    w1        = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3    = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4    = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7    = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    '''
    
    1x1 convolution of vgg encoder layer 7 as above
    The pretrained VGG-16 model is already fully convolutionalized, i.e. it
    already contains the 1x1 convolutions that replace the fully connected
    layers. THOSE 1x1 convolutions are the ones that are used to preserve
    spatial information that would be lost if we kept the fully connected
    layers. The purpose of the 1x1 convolutions that we are adding on top
    of the VGG is merely to reduce the number of filters from 4096 to the
    number of classes which 2 for our model.
    
    '''
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1),
                        padding='same',
                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    '''
    conv_1x1 = tf.Print(conv_1x1, [tf.shape(conv_1x1)], summarize=6,
                        name="vgg7conv_1x1")
    '''

    '''
    3# Upsample, input shape [x x1 x2 x3] output shape [x x1*2 x2*2 x3]
    '''
    output_32 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, strides=(2, 2),
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    '''
    output_32 = tf.Print(output_32, [tf.shape(output_32)], summarize=6,
                         name="First_Upsample")
    '''

    '''
    4# 1x1 convolution of vgg pool4 layer to match the shape with above upsample layer
    output shape [x x1*2 x2*2 x3]
    '''
    vgg_layer4_out_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,
                 strides=(1,1), padding='same',
                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    '''
    vgg_layer4_out_1x1 = tf.Print(vgg_layer4_out_1x1 , [tf.shape(vgg_layer4_out_1x1)],
                                   summarize=6, name="pool4conv1x1")
    '''

    '''
    5# skip connection, matrix element-wise addition
    input = output = [x x1*2 x2*2 x3]
    '''
    output_32_skip = tf.add(output_32, vgg_layer4_out_1x1)

    '''
    output_32_skip = tf.Print(output_32_skip, [tf.shape(output_32_skip)],
                              summarize=6, name="First_Skip_layer_with_pool4")
    '''
    '''
    6# Second Upsample, input [x x1*2 x2*2 x3] output [x x1*2*2 x2*2*2 x3] 
    '''
    output_16 = tf.layers.conv2d_transpose(output_32_skip, num_classes, 4,
                strides=(2, 2), padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    '''
    output_16 = tf.Print(output_16, [tf.shape(output_16)], summarize=6,
                         name="Second_Upsample")
    '''

    '''
    7# 1x1 convolution of vgg pool3 layer to match the shape with above
       upsample layer,output shape:[5 20 72 2]
    '''

    vgg_layer3_out_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,
                strides=(1,1), padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    '''
    vgg_layer3_out_1x1 = tf.Print(vgg_layer3_out_1x1,
                         [tf.shape(vgg_layer3_out_1x1)], summarize=6,
                         name="pool3_conv1x1")
    '''

    '''
    8# skip connection, matrix element-wise addition
    input = output shape [x x1*2*2 x2*2*2 x3]
    '''
    output_16_skip = tf.add(output_16, vgg_layer3_out_1x1)

    '''
    output_16_skip = tf.Print(output_16_skip, [tf.shape(output_16_skip)],
                                summarize=6, name="Second_Skip_with_pool3")
    '''

    '''
    9# Final upsample by 8x8 stride
    input shape [x x1*2*2 x2*2*2 x3], output shape [x x1*2*2*8 x2*2*2*8 x3]
    '''
    output_8 = tf.layers.conv2d_transpose(output_16_skip, num_classes, 16,
                strides=(8, 8),
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    '''
    output_8 = tf.Print(output_8, [tf.shape(output_8)], summarize=6,
                        name="Final_layer_FCN-8")
    '''

    return output_8
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # Reshape the label same as logits 
    label_reshaped = tf.reshape(correct_label, (-1,num_classes))

    # Converting the 4D tensor to 2D tensor. logits is now a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_reshaped))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-3
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    train_op = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)    
    
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    print('Training....\n')

    for epoch in range(epochs):
        trainingloss_per_epoch = []
        print("Epoch {}".format(epoch))
        for image, label in get_batches_fn(batch_size):
            #print("Batch {} image {} label {}".format(batch_size, len(image), len(label)))
            _, loss = sess.run([train_op, cross_entropy_loss],
            feed_dict={input_image:image, correct_label:label,
                               keep_prob:0.5, learning_rate:0.0001})
            trainingloss_per_epoch.append("{:.3}".format(loss))
        print("Epoch {} Loss {} \n".format(epoch+1, trainingloss_per_epoch))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_img, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)

        #4D tensor
        correct_label = tf.placeholder(tf.int32,
                                        shape=(None, None, None, num_classes),
                                        name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)     
        
        epochs = 50 #30#20#10
        batch_size = 5
        saver = tf.train.Saver()
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_img, correct_label, keep_prob,
                 learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_img)
        
        #save model
        saver.save(sess, './runs/udacity_semantic_seg_model.ckpt')
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
