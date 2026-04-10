#!/usr/bin/env python3
# Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag




def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np




def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict


#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return




# Path Planner
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_settings_filename, goal_point, stopping_dist):
        print("-" * 30) ##added during testing to make it easier to find the start of the planner in the terminal output
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_settings_filename)


        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]


        # Global Bounding Box (so it doesn't sample outside the maze - added during testing to speed up sampling)
        self.min_x, self.max_x = -1.0, 44.0
        self.min_y, self.max_y = -46.0, 5.0


        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.6 #m/s (Feel free to change!) ##changed
        self.rot_vel_max = 0.5 #rad/s (Feel free to change!) ##changed


        # Goal Parameters
        self.goal_point = goal_point
        self.stopping_dist = stopping_dist


        # Trajectory Simulation Parameters
        self.timestep = 1.0  
        self.num_substeps = 10


        # Planning storage (Start at 0,0,0)
        self.nodes = [Node(np.zeros((3, 1)), -1, 0)]
       
        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5


        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)        
        return


    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")


        ## Added some bias to sample in which direction to expand the tree
        # Goal Bias 20% of time - look in direction of the goal to speed up convergence
        if np.random.rand() < 0.20:
            return self.goal_point
       
        # Adaptive Local Window 60% of time - look only in a local window around the best node to speed up convergence while still allowing some exploration
        if np.random.rand() < 0.60:
            dists_to_goal = [np.linalg.norm(n.point[0:2] - self.goal_point) for n in self.nodes]
            best_pos = self.nodes[np.argmin(dists_to_goal)].point[0:2].flatten()
           
            # Closer to goal = smaller window
            dist_to_goal_val = np.min(dists_to_goal)
            window_size = 5.0 if dist_to_goal_val < 5.0 else 20.0
           
            x = np.random.uniform(max(self.min_x, best_pos[0] - window_size/2),
                                  min(self.max_x, best_pos[0] + window_size/2))
            y = np.random.uniform(max(self.min_y, best_pos[1] - window_size/2),
                                  min(self.max_y, best_pos[1] + window_size/2))
            return np.array([[x], [y]])


        # Global Random Exploration 20% - sample anywhere in the map to ensure we don't miss any narrow passages
        x = np.random.uniform(self.min_x, self.max_x)
        y = np.random.uniform(self.min_y, self.max_y)
        return np.array([[x], [y]])


    def check_if_duplicate(self, point):
        # #Check if point is a duplicate of an already existing node
        # print("TO DO: Check that nodes are not duplicates")


        if not self.nodes:
            return False


        # Calculate the distance to nearest neighbor
        idx = self.closest_node(point)
        closest_coords = self.nodes[idx].point[0:2]
        dist = np.linalg.norm(closest_coords - point)
        if dist < 1e-3:
            return True
        else:
            return False


    def closest_node(self, point):
        # #Returns the index of the closest node
        # print("TO DO: Implement a method to get the closest node to a sapled point")


        if not self.nodes:
            return None
       
        dists = []
        for node in self.nodes:
            # Calculate distance using only X and Y (indices 0 and 1)
            node_coords = node.point[0:2]
            d = np.linalg.norm(node_coords - point)
            dists.append(d)
           
        return np.argmin(dists)




    def simulate_trajectory(self, node_i, point_s):
        # #Simulates the non-holonomic motion of the robot.
        # #This function drives the robot from node_i towards point_s. This function does has many solutions!
        # #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        # #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        vel, rot_vel = self.robot_controller(node_i, point_s)


        robot_traj = self.trajectory_rollout(vel, rot_vel, node_i)
        return robot_traj


    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        # return 0, 0
       
        xi, yi, thi = node_i.flatten()
   
        # Calculate distance and angle to target
        dx = point_s[0, 0] - xi
        dy = point_s[1, 0] - yi
        dist = np.hypot(dx, dy)
       
        # Determine the steering error (alpha)
        target_th = np.arctan2(dy, dx)
        alpha = (target_th - thi + np.pi) % (2 * np.pi) - np.pi
       
        # Set velocities based on distance and angle
        vel = max(0.1, min(self.vel_max, 0.5 * dist))
        rot_vel = np.clip(3.0 * alpha, -self.rot_vel_max, self.rot_vel_max)
       
        return vel, rot_vel
   
    def trajectory_rollout(self, vel, rot_vel, node_i):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        # return np.zeros((3, self.num_substeps))
       
        traj = np.zeros((3, self.num_substeps))
        dt = self.timestep / self.num_substeps
        curr = node_i.flatten().astype(float)
       
        for i in range(self.num_substeps):
            # Update X, Y, and Theta (Heading)
            curr[0] += vel * np.cos(curr[2]) * dt
            curr[1] += vel * np.sin(curr[2]) * dt
            curr[2] += rot_vel * dt
           
            traj[:, i] = curr
           
        return traj
   
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        # return 0


        res = self.map_settings_dict["resolution"]
        orig = self.map_settings_dict["origin"]
        point = np.array(point)
       
        x = point[0, :]
        y = point[1, :]
       
        # Convert to Pixel Coordinates, col corresponds to x, row corresponds to y (inverted)
        col = ((x - orig[0]) / res).astype(int)
        row = (self.map_shape[0] - 1 - (y - orig[1]) / res).astype(int)
       
        # Bounds
        col = np.clip(col, 0, self.map_shape[1] - 1)
        row = np.clip(row, 0, self.map_shape[0] - 1)
       
        return row, col


    def points_to_robot_circle(self, points):
        # #Convert a series of [x,y] points to robot map footprints for collision detection
        # #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")
       
        res = self.map_settings_dict["resolution"]
        rows, cols = self.point_to_cell(points)
        radius_px = int(np.ceil(self.robot_radius / res))
        all_rows, all_cols = [], []
       
        for r, c in zip(rows, cols):
            # Check if point is inside map bounds
            if 0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1]:
                # Generate indices of a disk centered at (r, c)
                rr, cc = disk((r, c), radius_px, shape=self.map_shape)
                all_rows.extend(rr)
                all_cols.extend(cc)
               
        return all_rows, all_cols
   
    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
   
    def connect_node_to_point(self, node_i, point_f):
        # #Given two nodes find the non-holonomic path that connects them
        # #Settings
        # #node is a 3 by 1 node
        # #point is a 2 by 1 point
        # print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return self.simulate_trajectory(node_i, point_f)


    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        p1 = trajectory_o[0:2, 0]
        p2 = trajectory_o[0:2, -1]
        return np.linalg.norm(p1 - p2)


    def update_children(self, node_id):
        # #Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")
       
        parent_node = self.nodes[node_id]
        for child_id in parent_node.children_ids:
            child_node = self.nodes[child_id]
            dist = np.linalg.norm(child_node.point[0:2] - parent_node.point[0:2])
            child_node.cost = parent_node.cost + dist
            self.update_children(child_id)
        return
   
   ### ADDED HELPER FUNCTION FOR RRT* TO CHECK IF A STRAIGHT LINE PATH IS COLLISION FREE (USED IN CHOOSE BEST PARENT AND REWIRE STEPS) ###
    def is_line_collision_free(self, p1, p2):
        # Ensure p1 and p2 are flat 1D arrays for norm and linspace
        p1_flat = p1.flatten()[0:2]
        p2_flat = p2.flatten()[0:2]
       
        dist = np.linalg.norm(p1_flat - p2_flat)
        res = self.map_settings_dict["resolution"]
        num_samples = int(dist / (res / 2))
       
        if num_samples < 2:
            return True
           
        # np.linspace here creates (num_samples, 2).
        # .T turns it into (2, num_samples) which points_to_robot_circle expects.
        points = np.linspace(p1_flat, p2_flat, num_samples).T
       
        rr, cc = self.points_to_robot_circle(points)
       
        # Check if any sampled circle hits an obstacle (0)
        if len(rr) > 0:
            if np.any(self.occupancy_map[rr, cc] == 0):
                return False
        return True
    ###################


    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
           
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()
            if self.check_if_duplicate(point):
                continue


            #Get the closest point
            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]


            #Simulate driving the robot towards the closest point
            trajectory = self.simulate_trajectory(closest_node.point, point)
           
            ########################################
            #Check for collisions
            # print("TO DO: Check for collisions and add safe points to list of nodes.")
            rr, cc = self.points_to_robot_circle(trajectory[0:2, :])
            if len(rr) == 0 or np.any(self.occupancy_map[rr, cc] == 0):
                continue # Collision detected


            ### ADD NODE
            new_pt = trajectory[:, -1].reshape(3, 1)            
            cost = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory)


            new_node = Node(new_pt, closest_node_id, cost)
            self.nodes.append(new_node)
            closest_node.children_ids.append(len(self.nodes) - 1)


            ### ADD VISUALIZATION
            self.window.add_line(closest_node.point[0:2].flatten(), new_pt[0:2].flatten(), color=(50, 50, 255))
           
            if i % 500 == 0:
                dist_to_goal = np.linalg.norm(new_pt[0:2] - self.goal_point)
                print(f"Iter {i} | Nodes: {len(self.nodes)} | Dist: {dist_to_goal:.2f}m")


            ##################################################
            # #Check if goal has been reached
            # print("TO DO: Check if at goal point.")
            if np.linalg.norm(new_pt[0:2] - self.goal_point) < self.stopping_dist:
                print(f"Goal reached at iteration {i}!")
                return self.nodes


        return self.nodes
   
   
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot              
        for i in range(50000): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()
            if self.check_if_duplicate(point):
                continue


            #Closest Node
            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]


            #Simulate trajectory
            trajectory = self.simulate_trajectory(closest_node.point, point)
           
            ################################
            # #Check for Collision
            # print("TO DO: Check for collision.")
            rr, cc = self.points_to_robot_circle(trajectory[0:2, :])
            if len(rr) == 0 or np.any(self.occupancy_map[rr, cc] == 0):
                continue


            new_pt = trajectory[:, -1].reshape(3, 1)
            dist_moved = self.cost_to_come(trajectory)


            ###FIND NEAR NEIGHBORS
            radius = self.ball_radius()
            near_ids = [idx for idx, node in enumerate(self.nodes)
                        if np.linalg.norm(node.point[0:2] - new_pt[0:2]) < radius]


            ### CHOOSE BEST PARENT
            best_parent_id = closest_node_id
            min_cost = closest_node.cost + dist_moved
           
            for near_id in near_ids:
                near_node = self.nodes[near_id]
                cost_to_near = near_node.cost + np.linalg.norm(new_pt[0:2] - near_node.point[0:2])
               
                if cost_to_near < min_cost:
                    if self.is_line_collision_free(near_node.point, new_pt):
                        best_parent_id = near_id
                        min_cost = cost_to_near
                       
            new_node = Node(new_pt, best_parent_id, min_cost)
            self.nodes.append(new_node)
            new_node_id = len(self.nodes) - 1
            self.nodes[best_parent_id].children_ids.append(new_node_id)
           
            ######################################
            # #Last node rewire
            # print("TO DO: Last node rewiring")
            # #Close node rewire
            # print("TO DO: Near point rewiring")
            for near_id in near_ids:
                near_node = self.nodes[near_id]
                dist_from_new = np.linalg.norm(near_node.point[0:2] - new_pt[0:2])
               
                if new_node.cost + dist_from_new < near_node.cost:
                    if self.is_line_collision_free(new_pt, near_node.point):
                        if near_node.parent_id != -1:
                            self.nodes[near_node.parent_id].children_ids.remove(near_id)
                       
                        near_node.parent_id = new_node_id
                        near_node.cost = new_node.cost + dist_from_new
                        new_node.children_ids.append(near_id)
                        self.update_children(near_id)
                        self.window.add_line(new_pt[0:2].flatten(), near_node.point[0:2].flatten(), color=(0, 255, 255))


            ### ADD VISUALIZATION
            self.window.add_line(self.nodes[best_parent_id].point[0:2].flatten(), new_pt[0:2].flatten(), color=(50, 50, 255))
           
            if i % 500 == 0:
                dist_to_goal = np.linalg.norm(new_pt[0:2] - self.goal_point)
                print(f"Iter {i} | Nodes: {len(self.nodes)} | Goal Dist: {dist_to_goal:.2f}m")


            #####################################################
            # #Check for early end
            # print("TO DO: Check for early end")
            if np.linalg.norm(new_pt[0:2] - self.goal_point) < self.stopping_dist:
                print(f"Goal reached at iteration {i}!")
                return self.nodes


    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path


def main():
    start_time = time.time()


    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"


    #robot information
    goal_point = np.array([[42.05], [-44]]) #m
    stopping_dist = 0.5 #m
   
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())
   
    path_list = path_planner.recover_path()
    if path_list:
        total_dist = 0
        for i in range(len(path_list) - 1):
            p1 = path_list[i][0:2]
            p2 = path_list[i+1][0:2]
            total_dist += np.linalg.norm(p1 - p2)
        print(f"Total Path Distance: {total_dist:.2f} meters")
       
        node_path_metric = np.hstack(path_list)
        np.save("shortest_path_rrt_star_CZ1.npy", node_path_metric)
        print(f"Path saved. Duration: {time.time() - start_time:.2f}s")
   
    # Keep visualizer open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        time.sleep(0.05)
    pygame.quit()




if __name__ == '__main__':
    main()
