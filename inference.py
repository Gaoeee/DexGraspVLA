# Python standard library imports
import argparse
import os
import select
import sys
import threading
import time
from datetime import datetime
import yaml

# Third-party library imports
import cv2
import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_tensor

# Computer vision model imports
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from segment_anything import sam_model_registry, SamPredictor

# Local imports
from controller.common.pytorch_util import dict_apply
from planner.dexgraspvla_planner import DexGraspVLAPlanner
from inference_utils.utils import (cubic_spline_interpolation_7d, get_start_command,
                           load_config, show_mask, update_array, encode_image_to_base64, 
                           preprocess_img, check_url,
                           timer, log)


# Register now resolver
def now_resolver(pattern: str):
    """Handle ${now:} time formatting"""
    return datetime.now().strftime(pattern)

# Register resolvers
OmegaConf.register_new_resolver("now", now_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


class RoboticsSystem:
    def __init__(self, args):
        self.args = args
        self.config = self.load_inference_config()
        self.running = True

        self.init_robot()
        print("init_robot done")
        self.init_robot_state()
        print("init_robot_state done")
        self.init_controller()
        print("init_controller done")
        self.init_planner()
        print("init_planner done")
        self.init_utils_and_data()
        print("init_utils_and_data done")
        self.init_threads()
        print("init_threads done")


    def load_inference_config(self):
        """Load system configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), 'inference_utils', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


    def init_robot(self):
        # Load DOF limits from config
        self.dof_lower_limits = self.config['robot']['dof_limits']['lower']
        self.dof_upper_limits = self.config['robot']['dof_limits']['upper']

        # Initialize hands
        self.left_hand = Hand(port=self.config['robot']['hands']['left']['port'])
        self.right_hand = Hand(port=self.config['robot']['hands']['right']['port'])

        # Initialize arms
        self.left_arm = ArmControl(ip=self.config['robot']['arms']['left']['ip'])
        self.right_arm = ArmControl(ip=self.config['robot']['arms']['right']['ip'])

        # Initialize cameras
        self.right_first_image_processor = RealSenseImage(self.config['cameras']['right_first']['sn'])
        self.third_image_processor = RealSenseImage(self.config['cameras']['third']['sn'])


    def init_robot_state(self):
        # Load initial positions from config
        self.left_init_qpos = self.config['robot']['arms']['left']['init_qpos']
        self.right_init_qpos = self.config['robot']['arms']['right']['init_qpos']
        self.right_placement_joint = np.array(self.config['robot']['arms']['right']['placement_joint'])
        self.right_return_medium_joint = np.array(self.config['robot']['arms']['right']['return_medium_joint'])

        # Initialize arms
        self.left_arm.robot.Clear_System_Err()
        self.left_init_qpos = np.clip(self.left_init_qpos, self.dof_lower_limits, self.dof_upper_limits)
        self.left_arm.move_joint(self.left_init_qpos, speed=10)

        self.right_arm.robot.Clear_System_Err()
        self.right_init_qpos = np.clip(self.right_init_qpos, self.dof_lower_limits, self.dof_upper_limits)
        self.right_arm.move_joint(self.right_init_qpos, speed=10)

        self.left_arm.robot.Clear_System_Err()
        self.right_arm.robot.Clear_System_Err()

        # Initialize hands
        self.set_left_hand_open()
        self.set_right_hand_open()
        self.left_hand.set_angles(self.target_left_hand_joint)
        self.right_hand.set_angles(self.target_right_hand_joint)

        self.target_right_arm_joint = np.array(self.right_init_qpos)[None, :]
        self.update_target = True
        self.in_post_run = True


    def init_controller(self):
        self.device = torch.device('cuda:0')

        main_config_path = os.path.join(os.path.dirname(__file__), 'controller', 'config', 'train_dexgraspvla_controller_workspace.yaml')
        task_config_path = os.path.join(os.path.dirname(__file__), 'controller', 'config', 'task', 'grasp.yaml')
        
        self.cfg = load_config(
            main_config_path=main_config_path,
            task_config_path=task_config_path
        )
        workspace = hydra.utils.get_class(self.cfg._target_)(self.cfg)
        self.policy = workspace.model
        self.policy.eval().to(self.device)

        # Initialize SAM
        sam_checkpoint = self.config['sam']['checkpoint']
        model_type = self.config['sam']['model_type']
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Initialize Cutie
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1


    def init_planner(self):
        if not self.args.manual:
            assert check_url(self.config["planner"]["base_url"]), "URL is not working!"
            print("URL checked")
            self.planner = DexGraspVLAPlanner(
                api_key=self.config["planner"]["api_key"], 
                base_url=self.config["planner"]["base_url"])


    def init_utils_and_data(self):
        self.record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', self.config['logging']['exp_name'])
        os.makedirs(self.record_dir, exist_ok=True)
        resolution = self.config['cameras']['right_first']['resolution']
        self.right_first_color_image_buffer = np.zeros((self.cfg.n_obs_steps, resolution[1], resolution[0], 3))
        self.third_color_image_buffer = np.zeros((self.cfg.n_obs_steps, resolution[1], resolution[0], 4))
        self.state_buffer = np.zeros((self.cfg.n_obs_steps, 13))
        self.height_threshold = 0


    def init_threads(self):
        # Start a background thread to handle keyboard input
        self.arm_control_thread = threading.Thread(target=self.low_level_control, daemon=True)
        self.arm_control_thread.start()

        self.hand_control_thread = threading.Thread(target=self.low_level_hand_control, daemon=True)
        self.hand_control_thread.start()

        self.camera_thread = threading.Thread(target=self.capture_images, daemon=True)
        self.camera_thread.start()

        self.keyboard_listener_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_listener_thread.start()

        if not self.args.manual:
            self.check_grasp_success_thread = threading.Thread(target=self.check_grasp_success, daemon=True)
            self.check_grasp_success_thread.start()

        self.monitor_execution_time_thread = threading.Thread(target=self.monitor_execution_time, daemon=True)
        self.monitor_execution_time_thread.start()


    def low_level_control(self):
        self.init_low_level_control = True
        while True:
            if self.update_target:
                if self.init_low_level_control:
                    arm_joint = self.right_arm.get_current_joint()
                    self.init_low_level_control = False
                else:
                    arm_joint = self.arm_traj_interp[self.low_level_target_joint_index]

                points_7d = np.array([
                    arm_joint,
                    self.target_right_arm_joint[0],
                ])

                interpolation_num = self.config['control']['arm_trajectory']['interpolation_num']

                self.arm_traj_interp = cubic_spline_interpolation_7d(points_7d, step=(1 / interpolation_num))
                self.low_level_target_joint_index = 0
                self.update_target = False

            if not self.in_post_run:
                self.right_arm.move_joint_CANFD(self.arm_traj_interp[self.low_level_target_joint_index])
            self.low_level_target_joint_index += 1
            time.sleep(0.01)

            if self.low_level_target_joint_index >= interpolation_num - 1:
                self.low_level_target_joint_index = interpolation_num - 1

            if not self.running:
                break


    def low_level_hand_control(self):
        while True:
            self.right_hand.set_angles(self.target_right_hand_joint)
            self.right_hand_joint = self.right_hand.get_angles()
            time.sleep(0.05)  # TODO: The delay here may need adjustment
            if not self.running:
                break


    def capture_images(self):
        while True:
            self.right_first_color_image = self.right_first_image_processor.capture_rgb_frame()
            self.third_color_image = self.third_image_processor.capture_rgb_frame()
            if not self.running:
                break


    def keyboard_listener(self):
        """Thread to listen for keyboard input"""
        while True:
            if not self.in_post_run:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    self.log("<Enter> detected, ending this episode.")

                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    head_img_path = os.path.join(self.img_dir, f"{timestamp}_head_image_finish.png")
                    cv2.imwrite(head_img_path, self.third_color_image)
                    self.log(f"Head camera image saved.")

                    wrist_img_path = os.path.join(self.img_dir, f"{timestamp}_wrist_image_finish.png")
                    cv2.imwrite(wrist_img_path, self.right_first_color_image)
                    self.log(f"Wrist camera image saved.")

                    self.in_post_run = True
            else:
                time.sleep(0.1)  # Avoid excessive CPU usage
            if not self.running:
                break


    def check_grasp_success(self):
        while True:
            if not self.in_post_run:
                current_pose = self.right_arm.get_current_pose()
                current_height = current_pose[2]
                if current_height > self.height_threshold:
                    base64_image = encode_image_to_base64(self.third_color_image[..., ::-1])
                    image_url = f"data:image/png;base64,{base64_image}"
                    with timer("check_grasp_success", self.log_file):
                        if self.planner.request_task(
                            frame_path=image_url,
                            task_name="check_grasp_success",
                        ):
                            self.in_post_run = True
                            self.log("Planner believes the task is done.")
            time.sleep(0.1)
            if not self.running:
                break


    def monitor_execution_time(self):
        """Thread to monitor task execution time"""
        max_duration = self.config['control']['monitor']['max_episode_duration']
        
        while True:
            if not self.in_post_run:
                elapsed_time = time.time() - self.episode_start
                if elapsed_time > max_duration:
                    print(f"Episode exceeded maximum duration of {max_duration}s")
                    self.in_post_run = True
            # Reduce CPU usage
            time.sleep(1)
            if not self.running:
                break


    def run(self):
        if self.args.manual:
            self.run_manual()
        else:
            self.run_planner()


    def run_manual(self):
        while True:
            if not get_start_command():
                print("Bye.")
                break

            # Setup
            self.time_step = 0
            self.init_episode(manual=True)

            # Receive bounging box and track mask
            bbox = self.mark_bbox_manual()
            self.initialize_sam_cutie(bbox)
            self.reset_flags()

            self.log("Controller starts executing the current instruction.")

            # Execute grasping
            while True:
                if self.in_post_run:
                    break
                state = self.get_state()
                mask = self.get_mask()
                obs = self.get_obs(state, mask)  # TODO: There might be misalignment between mask and image, needs verification
                attn_map_output_path = os.path.join(self.current_attn_maps_dir, f'{self.time_step}.pkl') if self.args.gen_attn_map else None
                self.time_step += 1
                obs_dict_np = self.process_obs(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                with torch.no_grad():
                    action_pred = self.policy.predict_action(obs_dict, attn_map_output_path)
                    action = action_pred[0].detach().to('cpu').numpy()
                self.execute_action(action)

            # Finish and reset
            self.close_episode()
            self.right_finish_reset()


    def run_planner(self):
        while True:
            if not get_start_command():
                print("Bye.")
                break

            # Setup
            self.time_step = 0
            self.init_episode(manual=False)
            self.process_user_prompt()
            while True:
                self.get_current_instruction()
                bbox = self.mark_bbox_planner()
                self.initialize_sam_cutie(bbox)
                self.reset_flags()

                self.log("Controller starts executing the current instruction.")
                # Execute grasping
                while True:
                    if self.in_post_run:
                        break
                    state = self.get_state()
                    mask = self.get_mask()
                    obs = self.get_obs(state, mask)  # TODO: There might be misalignment between mask and image, needs verification
                    attn_map_output_path = os.path.join(self.current_attn_maps_dir, f'{self.time_step}.pkl') if self.args.gen_attn_map else None
                    self.time_step += 1
                    obs_dict_np = self.process_obs(env_obs=obs, shape_meta=self.cfg.task.shape_meta)
                    obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                    with torch.no_grad():
                        action_pred = self.policy.predict_action(obs_dict, attn_map_output_path)
                        action = action_pred[0].detach().to('cpu').numpy()
                    self.execute_action(action)
                
                self.right_finish_reset()
                user_prompt_complete = self.check_complete()
                if user_prompt_complete:
                    break
            self.close_episode()


    def init_episode(self, manual=False):
        # Create episode directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_episode_dir = os.path.join(self.record_dir, timestamp)
        os.makedirs(self.current_episode_dir, exist_ok=True)

        # Create log file
        self.log_file_path = os.path.join(self.current_episode_dir, 'log.txt')
        self.log_file = open(self.log_file_path, 'w')

        if manual:
            self.log(f"Episode starts, using manual mode.")
        else:
            self.log(f"Episode starts, using planner mode.")

        # Create image directory
        self.img_dir = os.path.join(self.current_episode_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        if not manual:
            self.planner.set_logging(self.log_file, self.img_dir)

        # Initialize data buffers and video writer
        if self.args.save_deployment_data:
            self.right_cam_buffer, self.rgbm_buffer, self.state_buffer_save, self.action_buffer = [], [], [], []

            width, height = 640, 480
            fps = 20
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.current_episode_dir, 'video.mp4')
            self.video_writer = cv2.VideoWriter(
                video_path,
                fourcc, fps, (width * 3, height))

        # Create attention maps directory
        if self.args.gen_attn_map:
            self.current_attn_maps_dir = os.path.join(self.current_episode_dir, 'attn_maps')
            os.makedirs(self.current_attn_maps_dir, exist_ok=True)


    def mark_bbox_manual(self):
        # Save original image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{timestamp}_head_image_start.png"
        img_path = os.path.join(self.img_dir, img_filename)
        cv2.imwrite(img_path, self.third_color_image)
        
        # Display image and get bounding box
        plt.figure()
        plt.imshow(self.third_color_image[..., ::-1])  # The image is BGR
        plt.axis('off')
        plt.title("Please click two points to define the bounding box (top left and bottom right)")
        bbox_points = plt.ginput(n=2, timeout=0)
        plt.close()
        (x1, y1), (x2, y2) = bbox_points
        bbox = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

        # Save image with bounding box
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_with_bbox_filename = f"{timestamp}_head_image_with_bbox.png"
        self.show_and_save_image_with_bbox(self.third_color_image[..., ::-1], bbox, img_with_bbox_filename)
        return bbox
    

    def initialize_sam_cutie(self, bbox):
        self.processor.clear_memory()
        torch.cuda.empty_cache()
        self.predictor.set_image(self.third_color_image[..., ::-1])

        masks, scores, _ = self.predictor.predict(box=bbox, multimask_output=True)
        self.best_mask = masks[np.argmax(scores)]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_with_mask_filename = f"{timestamp}_head_image_with_mask.png"
        self.show_and_save_image_with_mask(self.third_color_image[..., ::-1], self.best_mask, img_with_mask_filename)

        # Reinitialize Cutie
        self.mask = torch.from_numpy(self.best_mask.astype('uint8')).cuda()
        self.objects = np.unique(self.best_mask.astype('uint8'))
        self.objects = self.objects[self.objects != 0].tolist()
        self.cutie_initialized = False  # Reset Cutie initialization flag


    def process_user_prompt(self):
        # Show head camera image
        plt.imshow(self.third_color_image[..., ::-1])
        plt.axis('off')
        plt.pause(1)

        # Save head camera image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{timestamp}_head_image_for_user_prompt.png"
        img_path = os.path.join(self.img_dir, img_filename)
        cv2.imwrite(img_path, self.third_color_image)

        sys.stdin.flush()
        self.user_prompt = input("""Please enter your instruction: (It can be an abstract instruction like 'clear the table' or a specific object to be grabbed, such as 'grasp the red cups and blue cookies')\n>>>  """)
        self.log(f"User prompt: {self.user_prompt}")

        plt.close()

        with timer("classify user prompt", self.log_file):
            self.user_prompt_type = self.planner.request_task(
                            task_name="classify_user_prompt",
                            instruction=self.user_prompt,
                            max_token=256
            )
        self.log(f"User prompt type: {self.user_prompt_type}.")
        if self.user_prompt_type == "TypeI":  # Explicitly specifies grabbing specific items
            image_url = self.prepare_head_image()
            with timer("decompose user prompt", self.log_file):
                self.object_list = self.planner.request_task(
                            frame_path=image_url,
                            task_name="decompose_user_prompt",
                            instruction=self.user_prompt,
                            max_token=512
                        )
                self.log(f"Object list: {self.object_list}.")
        elif self.user_prompt_type == "TypeII":  # Abstract instruction without specific details
            pass
        else:
            raise ValueError(f"The user prompt type {self.user_prompt_type} is invalid.")


    def get_current_instruction(self):
        if self.user_prompt_type == "TypeI":  # Explicitly specifies grabbing specific items
            self.current_instruction = self.object_list[0]
        elif self.user_prompt_type == "TypeII":  # Abstract instruction without specific details
            image_url = self.prepare_head_image()
            with timer("generate instruction", self.log_file):
                self.current_instruction = self.planner.request_task(
                            frame_path=image_url,
                            task_name="generate_instruction",
                            instruction=None,
                            max_token=512
                        )
        self.log(f"Current instruction: {self.current_instruction}")


    def mark_bbox_planner(self):
        image_url = self.prepare_head_image()
        with timer("mark bounding box", self.log_file):
            bbox_info = self.planner.request_task(
                    frame_path=image_url,
                    task_name="mark_bounding_box",
                    instruction=self.current_instruction
                )
        bbox = bbox_info['bbox_2d']
        self.log(f"Bounding box marked by the planner: {bbox}.")
        bbox = np.array(bbox)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_with_bbox_filename = f"{timestamp}_head_image_with_bbox.png"
        self.show_and_save_image_with_bbox(self.third_color_image[..., ::-1], bbox, img_with_bbox_filename)
        return bbox


    def check_complete(self):
        image_url = self.prepare_head_image()
        with timer("check instruction complete", self.log_file):
            self.current_instruction_complete = self.planner.request_task(
                            frame_path=image_url,
                            task_name="check_instruction_complete",
                            instruction=self.current_instruction
                        )
        if self.current_instruction_complete:
            self.log(f"Current instruction <{self.current_instruction}> is completed.")
            if self.user_prompt_type == "TypeI":
                self.object_list = self.object_list[1:]
                user_prompt_complete = len(self.object_list) == 0
            else:
                with timer("check user prompt complete", self.log_file):
                    user_prompt_complete = self.planner.request_task(
                                    frame_path=image_url,
                                    task_name="check_user_prompt_complete",
                                    instruction=None
                                )
        else:
            self.log(f"Current instruction <{self.current_instruction}> is not completed.")
            user_prompt_complete = False
        if user_prompt_complete:
            self.log(f"User prompt <{self.user_prompt}> is completed.")
        else:
            self.log(f"User prompt <{self.user_prompt}> is not completed.")
        return user_prompt_complete


    def show_and_save_image_with_bbox(self, image, bbox, filename):
        """Save and display image with bounding box
        
        Args:
            image: Original image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            filename: Filename to save
        """
        # Create an image with the same size as the original
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Create axes without margins
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Display image
        ax.imshow(image)
        
        # Get bounding box color and line width from config
        bbox_color = self.config['visualization']['bbox']['color']
        bbox_linewidth = self.config['visualization']['bbox']['linewidth']
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox.astype(int)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=bbox_linewidth, 
                            edgecolor=bbox_color, 
                            facecolor='none')
        ax.add_patch(rect)
        
        # Save image
        img_path = os.path.join(self.img_dir, filename)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        
        # Display image
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        
        self.log(f"Head camera image with bounding box saved.")


    def show_and_save_image_with_mask(self, image, mask, filename):
        """Save and display image with mask
        
        Args:
            image: Original image
            mask: Binary mask
            filename: Filename to save
        """
        # Create an image with the same size as the original
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Create axes without margins
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Display image
        ax.imshow(image)
        
        # Get mask color and random color settings from config
        mask_color = self.config['visualization']['mask']['color']
        
        # Display mask with configured color
        show_mask(mask, ax, color=mask_color)
        
        # Save image
        img_path = os.path.join(self.img_dir, filename)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        
        # Display image
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        
        self.log(f"Head camera image with mask saved.")


    def get_state(self):
        right_hand_joint = self.right_hand_joint.copy()
        right_arm_joint = self.unscale(self.right_arm.get_current_joint(),
                                                        self.dof_lower_limits[:7], 
                                                        self.dof_upper_limits[:7])
        state = np.concatenate([right_arm_joint, right_hand_joint])
        return state


    def get_mask(self):
        # Update Cutie mask
        with torch.no_grad():
            image_tensor = to_tensor(self.third_color_image[..., ::-1].copy()).cuda().float()
            if self.cutie_initialized == False:
                output_prob = self.processor.step(image_tensor, self.mask, objects=self.objects)
                self.cutie_initialized = True
            else:
                output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)
        return current_mask_np


    def get_obs(self, state, mask):
        # Update image buffer
        self.third_color_image_with_mask = np.concatenate([
            self.third_color_image,
            mask[..., None]
        ], axis=-1)
        self.right_first_color_image_buffer = update_array(
            self.right_first_color_image_buffer, 
            self.right_first_color_image.copy()
        )
        self.third_color_image_buffer = update_array(
            self.third_color_image_buffer, 
            self.third_color_image_with_mask
        )
        self.state_buffer = update_array(self.state_buffer, state)
        obs = {"right_cam_img": self.right_first_color_image_buffer, "rgbm": self.third_color_image_buffer, "right_state": self.state_buffer}
        return obs
    

    def process_obs(self, env_obs, shape_meta):
        """Get observation dictionary, using torch for image processing"""
        obs_dict_np = {}
        obs_shape_meta = shape_meta['obs']
        
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            shape = attr.get('shape')

            if type == 'rgb':
                imgs_in = env_obs[key]
                rgb = torch.from_numpy(imgs_in[..., :3]).float()  # [T, H, W, 3]
                rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]
                # Scale image
                rgb = F.interpolate(
                    rgb / 255.0,
                    size=(shape[1], shape[2]),
                    mode='bilinear',
                    align_corners=False
                )
                obs_dict_np[key] = rgb.numpy()

            elif type == 'rgbm':  # Process mask image
                imgs_in = env_obs[key]
                # Convert to torch tensor and adjust dimensions
                rgb = torch.from_numpy(imgs_in[..., :3]).float()  # [T, H, W, 3]
                mask = torch.from_numpy(imgs_in[..., 3:]).float()
                # Adjust channel order
                rgb = rgb.permute(0, 3, 1, 2)  # [T, 3, H, W]
                # Scale RGB
                rgb = F.interpolate(
                    rgb / 255.0,
                    size=(shape[1], shape[2]),  # Use the size specified in shape_meta
                    mode='bilinear',
                    align_corners=False
                )
                # Process mask
                mask = mask.permute(0, 3, 1, 2)  # [T, 1, H, W]
                mask = F.interpolate(
                    mask,
                    size=(shape[1], shape[2]),
                    mode='nearest'
                )
                mask = (mask > 0.5).float()
                # Combine RGB and mask
                out_imgs = torch.cat([rgb, mask], dim=1)  # [T, 4, H, W]
                obs_dict_np[key] = out_imgs.numpy()

            elif type == 'low_dim':
                obs_dict_np[key] = env_obs[key].astype(np.float32)
        
        return obs_dict_np


    def execute_action(self, action):
        for k in range(6):
            state = self.get_state()
            mask = self.get_mask()
            self.get_obs(state, mask)

            # Execute action
            self.target_right_arm_joint = self.scale(action[k:k+1, 0:7], self.dof_lower_limits[:7], self.dof_upper_limits[:7])
            self.update_target = True
            self.target_right_hand_joint = self.right_hand.execute_action(action[k, 7:])

            # Save data
            if self.args.save_deployment_data:
                self.record_frame(
                    state=state,
                    action=action[k],
                    right_cam_img=self.right_first_color_image.copy(),
                    rgbm=self.third_color_image_with_mask
                )


    def record_frame(self, state, action, right_cam_img, rgbm):
        """Record single frame data to buffer"""
        # Add to buffer
        self.right_cam_buffer.append(right_cam_img)
        self.rgbm_buffer.append(rgbm)
        self.state_buffer_save.append(state)
        self.action_buffer.append(action)
        
        # Prepare mask image
        mask_img = rgbm[..., 3]  # Get mask channel
        mask_colored = np.zeros_like(right_cam_img)  # Create colored mask
        mask_colored[mask_img > 0] = [0, 255, 0]  # Mark mask area as green
        
        # Create combined image and write
        combined_frame = np.hstack([
            right_cam_img,  # Right camera image
            rgbm[..., :3],  # RGB part of RGBM
            mask_colored    # Colored mask
        ])
        self.video_writer.write(combined_frame)


    def scale(self, x, lower, upper):
        """
        Scale a list of angles from [-1, 1] to a specified range [lower, upper].

        Parameters:
        x (list or np.ndarray): List of angles in the range [-1, 1].
        lower (list or np.ndarray): Lower bounds for each angle.
        upper (list or np.ndarray): Upper bounds for each angle.

        Returns:
        np.ndarray: Scaled angles.
        """
        x = np.array(x, dtype=np.float32)
        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)

        if len(x.shape) > 1:
            lower = np.expand_dims(lower, axis=0)  # Shape (1, 7)
            upper = np.expand_dims(upper, axis=0)  # Shape (1, 7)

        return 0.5 * (x + 1.0) * (upper - lower) + lower


    def unscale(self, x, lower, upper):
        """
        Convert a list of scaled values to [-1, 1].

        Parameters:
        x (list of float): List of scaled values.
        lower (list of float): List of lower bounds for scaling.
        upper (list of float): List of upper bounds for scaling.
        
        Returns:
        list of float: List of unscaled values.
        """
        return [(xi - lower_i) * 2 / (upper_i - lower_i) - 1 for xi, lower_i, upper_i in zip(x, lower, upper)]


    def set_left_hand_open(self):
        self.target_left_hand_joint = np.array(self.config['robot']['hands']['left']['default_open'])


    def set_right_hand_open(self):
        self.target_right_hand_joint = np.array(self.config['robot']['hands']['right']['default_open'])


    def right_finish_reset(self):
        self.move_to_target_joint_angle(self.right_placement_joint, step=0.05)
        self.set_left_hand_open()
        self.set_right_hand_open()
        self.move_to_target_joint_angle(self.right_return_medium_joint, step=0.05)
        self.move_to_target_joint_angle(self.right_init_qpos, step=0.05)


    def move_to_target_joint_angle(self, target_joint_angle, step=None):
        if step is None:
            step = self.config['control']['arm_trajectory']['interpolation_step']
        
        points_7d = np.array([
            self.right_arm.get_current_joint(),
            target_joint_angle,
        ])
        arm_traj_interp = cubic_spline_interpolation_7d(points_7d, step=step)
        for joint in arm_traj_interp:
            self.right_arm.move_joint_CANFD(joint)
            time.sleep(0.01)

        error_threshold = self.config['control']['arm_trajectory']['position_error_threshold']
        while True:
            self.right_arm.move_joint_CANFD(arm_traj_interp[-1])
            error_right = np.max(np.abs(np.array(self.right_arm.get_current_joint()) - arm_traj_interp[-1]))
            if error_right < error_threshold:
                break
            time.sleep(0.01)


    def reset_flags(self):
        self.init_low_level_control = True
        self.target_right_arm_joint = np.array(self.right_init_qpos)[None, :]
        self.update_target = True
        self.in_post_run = False
        self.episode_start = time.time()

    
    def prepare_head_image(self):
        processed_image = preprocess_img(self.third_color_image[..., ::-1])
        base64_image = encode_image_to_base64(processed_image)
        image_url = f"data:image/png;base64,{base64_image}"
        return image_url


    def log(self, message):
        log(message, self.log_file)


    def close_episode(self):
        """Save data for the entire episode"""
        if self.args.save_deployment_data:
            # Convert lists to numpy arrays
            right_cam_array = np.array(self.right_cam_buffer)
            rgbm_array = np.array(self.rgbm_buffer)
            state_array = np.array(self.state_buffer_save)
            action_array = np.array(self.action_buffer)
            
            # Save h5 file in the current episode directory
            h5_path = os.path.join(self.current_episode_dir, "data.h5")
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('right_cam_img', data=right_cam_array)
                f.create_dataset('rgbm', data=rgbm_array)
                f.create_dataset('state', data=state_array)
                f.create_dataset('action', data=action_array)

            self.log(f"Total number of frames in this episode: {len(self.right_cam_buffer)}.")
            
            # Close video writer and create new one
            self.video_writer.release()
        
        self.log("Episode ends.")

        self.log_file.close()


    def close(self):
        try:
            # First set flag to stop all threads
            self.running = False
            
            # Wait for all threads to end
            if hasattr(self, 'arm_control_thread'):
                self.arm_control_thread.join(timeout=2)
            if hasattr(self, 'hand_control_thread'):
                self.hand_control_thread.join(timeout=2)
            if hasattr(self, 'camera_thread'):
                self.camera_thread.join(timeout=2)
            if hasattr(self, 'keyboard_listener_thread'):
                self.keyboard_listener_thread.join(timeout=2)
            if hasattr(self, 'check_grasp_success_thread'):
                self.check_grasp_success_thread.join(timeout=2)
            if hasattr(self, 'monitor_execution_time_thread'):
                self.monitor_execution_time_thread.join(timeout=2)

            self.right_first_image_processor.close()
            self.third_image_processor.close()

            # Finally close robot devices
            self.set_left_hand_open()
            self.set_right_hand_open()
            self.left_hand.close()
            self.right_hand.close()
            self.left_arm.close()
            self.right_arm.close()

        except Exception as e:
            print(f"Error during close: {str(e)}")
        finally:
            print("System shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DexGraspVLA Inference')
    parser.add_argument('--manual', action='store_true', help='Manually select the target object.')
    parser.add_argument('--gen_attn_map', action='store_true', help='Generate attention maps.')
    parser.add_argument('--save_deployment_data', action='store_true', help='Save deployment data.')
    args = parser.parse_args()
    system = RoboticsSystem(args)
    system.run()
    system.close()
