from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

ForSIM = r'/light_classification/sim_frozen_inference_graph.pb'
ForREAL = r'/light_classification/real_frozen_inference_graph.pb'
class TLClassifier(object):
    def __init__(self, Simulation):
        #TODO load classifier
        #setup a new graph
        if Simulation:
            Model_Path = ForSIM
        else:
            Model_Path = ForReal
        self.graph = tf.Graph()
        #set the graph as default graph
        with self.graph.as_default():
            with tf.gfile.GFile(Model_Path, 'rb') as f:
                temp_graph_def = tf.GraphDef() #setup a temporary graph to contain default graph
                temp_graph_def.ParseFromString(f.read())
                tf.import_graph_def(temp_graph_def, name='')
            self.input_image = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.detections = self.graph.get_tensor_by_name('num_detections:0')
		
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        with self.graph.as_default():
            input_expand = np.expand_dims(image, axis=0) #input tensor placeholder has shape like [None, height, width, 3]
            boxes, scores, classes, detections = self.sess.run([self.boxes,self.scores,self.classes,self.detections], feed_dict={self.input_image:input_expand})
		
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
		
        if scores[0] > 0.5:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN
