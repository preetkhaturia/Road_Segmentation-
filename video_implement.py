
import tensorflow as tf
import scipy
import numpy as np
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class ProcessImage:
    def __init__(self, sess):
        self.sess = sess
    def __call__(self, image):
     
        image = scipy.misc.imresize(image, (256,256)) ##same size which is used while training the model
        image = (image.astype(float))
        
        graph = tf.get_default_graph()
        X_infer = graph.get_tensor_by_name('image_input:0')
        logits = graph.get_tensor_by_name('logits:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        image_shape = (256,256)
        out = sess.run(
        [tf.nn.softmax(logits)],
            {keep_prob: 1.0, X_infer: [image]})
        
        im_softmax = out[0][:, 1].reshape(image_shape[0], image_shape[1])
      
        segmentation = (im_softmax < 0.5).reshape(image_shape[0], image_shape[1], 1)
        
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        
        mask = scipy.misc.toimage(mask, mode="RGBA")
        
        street_im = scipy.misc.toimage(image)
        
        street_im.paste(mask, box=None, mask=mask)
        
        return np.array(street_im)
        


#sess=tf.Session()    
path = os.getcwd()
sess = tf.Session()
saver = tf.train.import_meta_graph(path+"/saver/-1000.meta")
saver.restore(sess,tf.train.latest_checkpoint(path+"/saver/"))
   

white_output = path+'/project_video_output.mp4'
### add your video here
clip1 = VideoFileClip(path+'/VID_test.mp4').subclip(3,15)  ## to test on whole video remove subclip
white_clip = clip1.fl_image(ProcessImage(sess))
white_clip.write_videofile(white_output, audio=False)


