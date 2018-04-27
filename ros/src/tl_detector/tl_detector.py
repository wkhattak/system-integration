#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import time
import numpy as np
import os

STATE_COUNT_THRESHOLD = 3
ROS_RATE = 15

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub4 = rospy.Subscriber('/image_color', Image, self.image_cb)
        # ########################## Instead use raw data for image classification ################################################

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
	self.sim = self.config["sim"]

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        #self.light_classifier = TLClassifier(self.sim)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.waypoints_2d = None
        self.waypoint_tree = None
		
	self.debug = self.config["debug"]
	if self.debug:
	    self.debug_file = open(os.path.join(os.getcwd(), 'debug.csv'), 'w')
	    self.debug_file.write('Image,Prediction,Ground Truth\n')
	    #rospy.logwarn('Current workig dir: %s', os.getcwd())
		
	#rospy.logwarn('self.waypoints_2d = %s', self.waypoints_2d)
        self.light_classifier = TLClassifier(self.sim)
        rospy.spin()
	#self.loop()

    def loop(self):
        rate = rospy.Rate(ROS_RATE)
        while not rospy.is_shutdown():
	    #self.image_cb_dummy()############################################################
	    rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
	#rospy.logwarn('traffic_cb called!!!')

    def image_cb_dummy(self):#####################################################################
	self.image_cb(None)
	
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
	#rospy.logwarn('Image dims --- Height: %s, Width: %s', msg.height, msg.width)
	#image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
	#cv2.imwrite('/home/student/CarND-Capstone-master/imgs/simulator/' + str(rospy.Time.now()) + '.png', image)
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
	    #rospy.logwarn('TL Publisher --- TL index: %s', light_wp)
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
	    #rospy.logwarn('TL Publisher --- TL index: %s', self.last_wp)
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]  
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
	result = self.light_classifier.get_classification(cv_image)
		
	if self.debug:
	    image_name = str(rospy.Time.now()) + '-' + str(result) + '.png'
	    img_path = os.getcwd() + '/../../../imgs/' + image_name
	    cv2.imwrite(img_path,  cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
	    self.debug_file.write(image_name + ',' + str(result) + ',' + str(light.state) + '\n')
        return result
        
        
        # ############################################### TESTING ONLY ##################################################################
        #return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                
                # Get closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            #rospy.logwarn('Traffic light state: %s', state)
            #rospy.logwarn('Closest traffic light stop line index: %s, state: %s', line_wp_idx, state)
            return line_wp_idx, state
        
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN
		
    def __del__(self):
        if self.debug:
	    self.debug_file.close()

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
