import numpy as np
import tensorflow as tf

with tf.Session() as sess:

    #Loading the meta graph
    new_saver = tf.train.import_meta_graph('mnist_model/save.ckpt.meta')

    #Loading the state of the graph
    new_saver.restore(sess, "mnist_model/save.ckpt")

    #Getting the default graph
    graph = tf.get_default_graph()

    print("\ntrainbles")
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in var:
        print(v.name, v.shape)
    print("\n")

    print("\nvariables")
    nodes = [n for n in graph.as_graph_def().node]
    for node in nodes:
        if("fcon3_layer" in node.name and ("Softmax" in node.name or "softmax" in node.name)):
            print(node.name, type(node))
    print("\n")

    print("\nplaceholders")
    placeholders = [ op for op in graph.get_operations() if op.type == "Placeholder"]
    for placeholder in placeholders:
        print(placeholder.name)
    print("\n")

    print("\nopperations")
    placeholders = [ op for op in graph.get_operations() if "Softmax" in op.name]
    for placeholder in placeholders:
        print(placeholder.name)
    print("\n")

    print("variables2\n")
    variables = tf.all_variables()
    for var in variables:
        print(var.name, var.shape)
    print("\n")

    print("\nvariables_action")
    variables = tf.all_variables()
    for var in variables:
        if("action" in var.name):
            print(var.name, var.shape)
    print("\n")

    test = tf.get_default_graph().get_tensor_by_name(u'clas/fcon3_layer/Softmax:0')
    print(type(test))
    print(test)
