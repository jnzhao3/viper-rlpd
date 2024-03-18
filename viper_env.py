from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import sys
import dm_env 
import cv2 

from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

# Instantiate CvBridge

# ROSPY IMPORTS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

from std_msgs.msg import Float64MultiArray
from Wiper import Wiper
# END ROSPY IMPORTS

HOME_POS = np.array([0.236494, 0, 0.42705])
VALID_LOC_LOW = np.array([0.2, -0.2, 0.25])
VALID_LOC_HIGH = np.array([0.4, 0.2, 0.6])

# GOALS = [np.array([0, .4]), np.array([-.2, .4]), np.array([.2, .4]), np.array([.1, .6])]
GOALS = [np.array([.1, .2])]
# bridge = CvBridge()

# VALID_LOC_LOW = np.array([-np.inf, -np.inf, -np.inf])
# VALID_LOC_HIGH = np.array([np.inf, np.inf, np.inf])

class ViperX(dm_env.Environment):

    def __init__(self, pixels=False):
        self._bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
        self._robot = self._bot.arm
        # self._robot.go_to_home_pose()
        
        self._ee_goal = np.array([0.0, 0.0])
        self.use_pixels = pixels
        if self.use_pixels:
            # USE BLACKED OUT IMAGES FOR NOW
            self.image_subscriber = rospy.Subscriber("/gripper_cam_yellow/mage_raw", Image, self._image_callback)
            # self.video = cv2.VideoCapture(1)
            # Check if the webcam is opened correctly
            # if not self.video.isOpened():
            #     raise IOError("Cannot open webcam")
            self._latest_raw_img = None
            self._latest_image = self._get_image()
        self._latest_torques = self._get_torques()
        self._latest_ee_pose = self._get_ee_pose()

        self._max_time_steps = 20
        self._current_time_step = 0

    def _image_callback(self, data):
        print("Received an image!")
        # try:
            # Convert your ROS Image message to OpenCV2
        # self._latest_raw_img = data
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if data.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(data.height, data.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=data.data)
        # If the byt order is different between the message and the system.
        if data.is_bigendian == (sys.byteorder == 'little'):
            self._latest_raw_img = image_opencv.byteswap().newbyteorder()
        # else:
        #     self._latest_raw_img = image_opencv
        # except e:
        #     print(e)
        # else:
        #     # Save your OpenCV2 image as a jpeg 
        #     cv2.imwrite('camera_image.jpeg', cv2_img)
    
    def _get_image(self):
        assert self.use_pixels
        # ret, frame = self.video.read()
        frame = self._latest_raw_img
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # frame = np.zeros((64, 64,3), np.uint8) # TEMPORARY
        self._latest_image = frame
        return self._latest_image
    
    def _get_torques(self):
        taus = self._bot.dxl.robot_get_joint_states().effort
        taus = np.asarray(taus)/650.0
        taus = np.clip(taus, -1, 1)
        self._latest_torques = taus
        return self._latest_torques

    def _get_ee_pose(self):
        T_sb =  self._robot.get_ee_pose()
        xyz = T_sb[:3, 3]
        # return xyz
        return xyz[1:] # size 2
    
    def go_home(self):
        T_new = np.array([[1, 0, 0, 0.336494],
            [0, 1, 0, 0],
            [0, 0, 1, 0.42705],
            [0, 0, 0, 1]])
        self._robot.set_ee_pose_matrix(T_new)
        
    def reset(self) -> dm_env.TimeStep:
        # self._robot.go_to_sleep_pose()
        # self._robot.go_to_home_pose()
        self.go_home()
        self._ee_goal = GOALS[np.random.choice(len(GOALS))]
        self._current_time_step = 0
        return dm_env.restart(self._observation())
    
    def _process_action(self, action):
        scaled_action = .05 * action
        # check whether the action is valid
        # if not, clip it to the valid range
        commanded_ee = self._latest_ee_pose + scaled_action
        if np.any(commanded_ee < VALID_LOC_LOW[1:]) or np.any(commanded_ee > VALID_LOC_HIGH[1:]):
            print("Action out of range, clipping")
            commanded_ee = np.clip(commanded_ee, VALID_LOC_LOW[1:], VALID_LOC_HIGH[1:])

        # return the clipped action
        return commanded_ee - self._latest_ee_pose

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = np.linalg.norm(achieved_goal - desired_goal)
        return np.exp(-5.0*d)

    def step(self, action) -> dm_env.TimeStep:
        self._current_time_step += 1 
        # action = np.array([0.0, action[0], action[1]])
        proc_action = self._process_action(action)
        # delta_x, delta_y, delta_z = proc_action[0], proc_action[1], proc_action[2]
        delta_y, delta_z = proc_action[0], proc_action[1]



        self._robot.set_ee_pose_components(
            x=self._robot.T_sb[0, 3],
            y=self._robot.T_sb[1, 3] + delta_y,
            z=self._robot.T_sb[2, 3] +delta_z,
            roll=0,
            pitch=0,
            yaw=0,
            moving_time=0.5,
            blocking=True,
        )

        #self._robot.set_ee_cartesian_trajectory(x=0.0, y=delta_y, z=delta_z)

        if self.use_pixels:
            self._get_image()
        self._get_torques()
        self._latest_ee_pose = self._get_ee_pose()

        reward = self.compute_reward(self._latest_ee_pose, self._ee_goal)




        if self._current_time_step >= self._max_time_steps:
            return dm_env.truncation(reward=reward, observation=self._observation())
        # elif reward > 0.95:
            # return dm_env.termination(reward=100.0, observation=self._observation())
        else:
            return  dm_env.transition(reward=reward, observation=self._observation())

    def observation_spec(self) -> dm_env.specs.BoundedArray:
        if self.use_pixels:
            image = dm_env.specs.BoundedArray(
                shape=(64, 64, 3), dtype=np.uint8, name='image',
                minimum=np.full((64, 64, 3), 0), maximum=np.full((64, 64, 3), 255))
        proprio = dm_env.specs.BoundedArray(
            shape=(2,), dtype=np.float32, name='ee_pose',
            minimum=np.array([-0.2, -0.2]), maximum=np.array([0.3,0.3]))
            # shape=(3,), dtype=np.float32, name='ee_pose',
            # minimum=VALID_LOC_LOW, maximum=VALID_LOC_HIGH)
        # torques = dm_env.specs.BoundedArray( # TODO: fix the actual min max values
            # shape=(9,), dtype=np.float32, name='torques',
            # minimum=np.full((9,), -1), maximum=np.full((9,), 1))
        goal = dm_env.specs.BoundedArray(
            shape=(2,), dtype=np.float32, name='goal',
            # minimum=VALID_LOC_LOW, maximum=VALID_LOC_HIGH)
            minimum=np.array([-0.1, -0.1]), maximum=np.array([0.1,0.1]))
        if self.use_pixels:
            return {
                'image': image, 
                'proprio': proprio, 
                # 'torques': torques,
                'goal': goal,
            }
        else:
            return dm_env.specs.BoundedArray(
                shape=(4,), dtype=np.float32, minimum=-np.inf, maximum=np.inf
            )
            # return {
            #     'proprio': proprio, 
            #     'torques': torques,
            #     'goal': goal,
            # }
    
    def action_spec(self) -> dm_env.specs.BoundedArray:
        # actions are a 2D vector
        return dm_env.specs.BoundedArray(
            shape=(2,), dtype=np.float32, name='action',
            minimum=-1, maximum=1)

    def _observation(self):
        if self.use_pixels:
            return {'image': self._latest_image, 
                'proprio': self._latest_ee_pose,
                # 'torques': self._latest_torques,
                'goal': self._ee_goal}
        else:
            return np.concatenate([self._latest_ee_pose, self._ee_goal])
            # return {'proprio': self._latest_ee_pose,
            #     # 'torques': self._latest_torques,
            #     'goal': self._ee_goal}
    
# WHITEBOARD MOTIONS

# def collect_image(data):
#     np_arr = np.fromstring(data.data, np.uint8)
#     image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     cv2.imshow("Image", image_np)
#     cv2.waitKey(1)

# def single_wipe(env, wipe_range):
#     env.step(np.array([0, 0, wipe_range]))
#     return -wipe_range

# TODO: fix this:
# STATUS = [] # 0 = not clean, 1 = clean

# def set_status(data):
#     STATUS.append(data.data)
#     print(data.data)


# def wiper(env, WIPE_RANGE):
#     # rospy.init_node('wiper', anonymous=True)

#     env.reset()
#     env.step(np.array([1, 0, 1]))
#     env.step(np.array([0, 0, -5]))
#     env.step(np.array([0, 0, 5]))

#     STATUS = 0

    # rospy.Subscriber("/cleanliness", String, set_status)

# END WHITEBOARD SPECIFIC MOTIONS


def main():
    images = []

    env = ViperX(pixels=False)
    env.reset()
    # import ipdb
    import ipdb; ipdb.set_trace()

    # rospy.Subscriber("/wiping_status", Float64MultiArray, env.step)
    # wiper()

    # while True:
        # wipe_range = single_wipe(env, wipe_range)
    # import ipdb; ipdb.set_trace()
    env.reset()
    # env.step(np.array([-1, 0, 0]))
    for _ in range(10):
        random_action = np.random.uniform(-.5, .5, size=(3,))
        random_action[0] = 0
        env.step(random_action)
        # env.step(np.array([.5, 0, 1]))
    # env.video.release() 
    # cv2.destroyAllWindows()

if __name__=='__main__':
    main()
   

# class ViperXServer

# localhost()

# class ViperX:
#     def init():
#         bot = InterbotixManipulatorXS("vx300s", "arm", "gripper")
#         cv2 acmp;

#     def step()
#         camer
#         return 

    


#python train_viper_reaching_pixels.py --env_name=viperx --start_training 100 \ --max_steps 300000 \ --config=configs/rlpd_pixels_config.py --offline_ratio=0 --eval_interval=5000 --save_video=True 
