import tensorflow as tf
 
sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

feed_dict={self.loss: loss / num_steps}
 
print(sess.run(op_to_restore,feed_dict))
#This will print 60 which is calculated 
#using new values of w1 and w2 and saved value of b1. 