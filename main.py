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
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # By scaling layers 3 and 4, the model learns much better (i.e. higher accuracy/IoU)
    # See https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
    
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    # First run 1x1 convolutions to collapse the number of channels/filters into just num_classes (i.e., 2)
    # conv1_7 has 512 channels/filters
    # Note the way we are regularizing here. We still need to add these regularization terms to the final loss
    # Not sure if padding = 'same' is needed given that we're doing 1x1 convolutions
    
    conv1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1_4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, kernel_size=1, padding='same',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv1_3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, kernel_size=1, padding='same',
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # strides of (2, 2) is what is doing the up-sampling here
    contrans1 = tf.layers.conv2d_transpose(conv1_7, num_classes, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    contrans_add1 = tf.add(contrans1, conv1_4)
    
    contrans2 = tf.layers.conv2d_transpose(contrans_add1, num_classes, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    contrans_add2 = tf.add(contrans2, conv1_3)
    
    contrans_output = tf.layers.conv2d_transpose(contrans_add2, num_classes, kernel_size=16, strides=(8, 8),
                                                 padding='same',
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return contrans_output
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
    
    # Convert to a 2D tensor where each row represents a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # why softmax, since there only 2 labels. One could just do a logistic regression.
    # Likely, this is because a softmax is more general
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label),
                                       name='fcn_cross_entropy_loss')
    # add all of the regularization terms that were introduced through the kernel_regularizer arguments
    regularization_term = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = cross_entropy_loss + regularization_term
    with tf.name_scope('summaries'):
        tf.summary.scalar('total_loss', total_loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    return logits, train_op, total_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, total_loss, input_image,
             correct_label, keep_prob, learning_rate, train_writer, merged, saver,
             learning_rate_val=0.001,
             keep_prob_val=0.5):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param total_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    overall_batch_nr = 0
    for epoch in epochs:
        batch_nr = 0
        loss_accumul = 0
        
        for image, label in get_batches_fn(batch_size):

            
            _ , loss, summary = sess.run([train_op, total_loss, merged],
                                feed_dict = {input_image: image,
                                             correct_label: label,
                                             learning_rate: learning_rate_val,
                                             keep_prob: keep_prob_val})
            loss_accumul += loss
            print("{} Epoch {}, batch {}, loss {}".format(str(datetime.datetime.now()), epoch + 1, batch_nr + 1, loss)) 
            
            batch_nr += 1
            overall_batch_nr += 1
            train_writer.add_summary(summary, overall_batch_nr)
        avg_loss = (loss_accumul / float(batch_nr))
        print("Epoch {}, average loss {}".format(epoch + 1, avg_loss))
        model_ckpt_name = "lr{}keep{}ep{}loss{}".format(str(learning_rate_val).replace(".", "_"),
                                                        str(keep_prob_val).replace(".", "_"),
                                                        epoch + 1,
                                                        str(avg_loss).replace(".", "_")) + ".ckpt"
        save_path = saver.save(sess, os.path.join("./models", model_ckpt_name))
        print("---------------------------------------")


def run(start_from=None):
    """
    Load data, train model, save trained model, execute model on test dataset
    :param start_from: Path to model checkpoint to start training from
    """
    num_classes = 2
    image_shape = (160, 576)
    epochs = range(20)
    batch_size = 8
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    tf.reset_default_graph()
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.float32, shape=None, name='correct_label')
        learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate') 
        
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        contrans_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, total_loss = optimize(contrans_output, correct_label, learning_rate, num_classes)
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./summaries/", sess.graph)
        saver = tf.train.Saver()
        
        if start_from is None:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, start_from)
            
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, total_loss, image_input,
                 correct_label, keep_prob, learning_rate, train_writer, merged, saver)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    run()
