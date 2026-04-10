#!/usr/bin/env python3
from __future__ import division, print_function
import os


import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros


# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker


# ros and se2 conversion utils
import utils




TRANS_GOAL_TOL = .3  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .5 # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0, 0.05, 0.15, 0.26] # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.82, 1.82, 11) # rad/s, max of real robot is 1.82
CONTROL_RATE = 5 # Hz, how frequently control signals are sent
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .05  # s, delta t to propagate trajectories forward by # Increased step size slightly to speed up rollout math
COLLISION_RADIUS = 0.225 # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = 0.05 # multiplier to change effect of rotational distance in choosing correct control # Lowered to prioritize forward progress
OBS_DIST_MULT = .1  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = 0.4  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'MYHAL_shortest_path_rrt_CZ.npy'


LOOK_AHEAD_DIST = 0.8 # m, how far ahead to look for the next node


# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
# TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions




class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()


        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running


        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)


        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)


        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        print(self.map_origin)
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        print(map)




        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5


        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()


        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))


        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'path.npy')).T
        # self.path_tuples = np.array(TEMP_HARDCODE_PATH)


        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)


        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0


        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)


        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT


        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))


        self.rate = rospy.Rate(CONTROL_RATE)


        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()


    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()


            self.update_pose()
            self.check_and_update_goal()


            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)


            # print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            # Grab linear (v) and angular (w) velocities from our predefined control options
            v = self.all_opts[:, 0]
            w = self.all_opts[:, 1]


            for t in range(1, self.horizon_timesteps + 1):
                # propogate trajectory forward, assuming perfect control of velocity and no dynamic effects
                # Step through time to predict where the robot goes for each v/w combination
                prev_pose = local_paths[t-1]
                theta = prev_pose[:, 2]
                # Update (x, y, theta) using the Unicycle Kinematics model
                local_paths[t, :, 0] = prev_pose[:, 0] + v * np.cos(theta) * INTEGRATION_DT
                local_paths[t, :, 1] = prev_pose[:, 1] + v * np.sin(theta) * INTEGRATION_DT
                local_paths[t, :, 2] = theta + w * INTEGRATION_DT




            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            map_h, map_w = self.map_np.shape
            valid_opts = []


            #print("TO DO: Check the points in local_path_pixels for collisions")
            # remove trajectories that were deemed to have collisions
            #print("TO DO: Remove trajectories with collisions!")
            for opt in range(self.num_opts):
                # Check the trajectory doesn't go outside the bounds of the map image
                traj_pix = local_paths_pixels[:, opt, :].astype(int)
                if np.any(traj_pix[:, 0] < 0) or np.any(traj_pix[:, 0] >= map_w) or \
                   np.any(traj_pix[:, 1] < 0) or np.any(traj_pix[:, 1] >= map_h):
                    continue
                # Check for obstacles -  values > 50 usually indicate a wall or obstacle
                if np.any(self.map_np[traj_pix[:, 1], traj_pix[:, 0]] > 50):
                    continue
                valid_opts.append(opt)




            # calculate final cost and choose best option
            # print("TO DO: Calculate the final cost and choose the best control option!")
            if not valid_opts:
                rospy.logwarn_throttle(1, "[WARN] All paths colliding! Safety Stop.")
                control = [0, 0]
            else:
                costs = []
                for opt in valid_opts:
                    # Look at the final predicted position of the trajectory
                    end_pose = local_paths[-1, opt]
                   
                    # Look at regular and angular distance
                    dist = np.linalg.norm(end_pose[:2] - self.cur_goal[:2])
                    abs_diff = np.abs(end_pose[2] - self.cur_goal[2])
                    rot_dist = min(np.pi * 2 - abs_diff, abs_diff)
                   
                   # Care more about linear distance, only care about angular distance when we are close to the node
                    cost = dist
                    if dist < MIN_TRANS_DIST_TO_USE_ROT:
                        cost += rot_dist * ROT_DIST_MULT
                    costs.append(cost)
               
                best_opt = valid_opts[np.argmin(costs)]
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))


            dist_to_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
            print(f"Targeting Node {self.cur_path_index} | Dist: {dist_to_goal:.2f}m | Safe Paths: {len(valid_opts)}/{self.num_opts} | Vel: {control[0]} m/s")


            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))
            self.rate.sleep()


    def update_pose(self):
        try:
            self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
            self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                      utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        except: pass




    def check_and_update_goal(self):
        # LOOK-AHEAD LOGIC: Scan path for furthest node within radius
        found_further = False
        for i in range(self.cur_path_index, len(self.path_tuples)):
            dist_to_node = np.linalg.norm(self.pose_in_map_np[:2] - self.path_tuples[i][:2])
            if dist_to_node < LOOK_AHEAD_DIST:
                if i > self.cur_path_index:
                    self.cur_path_index = i
                    self.cur_goal = np.array(self.path_tuples[i])
                    found_further = True
       
        if found_further:
            print(f">>> SKIPPED AHEAD to Node {self.cur_path_index}")




        # Completion check
        dist_from_final = np.linalg.norm(self.pose_in_map_np[:2] - self.path_tuples[-1][:2])
        if dist_from_final < TRANS_GOAL_TOL:
            print(f"\n[SUCCESS] Final waypoint reached within {TRANS_GOAL_TOL}m threshold.")
            rospy.loginfo("FINAL GOAL REACHED!")
            rospy.signal_shutdown("Path Complete")




    def stop_robot_on_shutdown(self):
        print("[SHUTDOWN] Stopping robot motors.")
        self.cmd_pub.publish(Twist())




if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.INFO)
        pf = PathFollower()
    except rospy.ROSInterruptException: pass