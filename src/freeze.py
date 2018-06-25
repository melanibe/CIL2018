import os

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
#code inspired from https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py

def freeze_graph(model_dir, checkpoint, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """   
    # We precise the file fullname of our freezed graph
    absolute_model_dir = model_dir
    output_graph = absolute_model_dir + "/frozen_model.pb"
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint), clear_devices=clear_devices)
        saver.restore(sess, checkpoint)
        # We use a built-in TF helper to export variables to constants
        with tf.variable_scope("discr") as scope:
            output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",")) # The output node names are used to select the usefull nodes 
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, absolute_model_dir + "/../frozen/frozen_model")
    return output_graph_def


