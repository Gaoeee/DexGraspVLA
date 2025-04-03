import httpx
import json_repair
from openai import OpenAI
import os
import time
import matplotlib.pyplot as plt
from planner.utils import parse_json, extract_list
from inference_utils.utils import decode_base64_to_image, log


class DexGraspVLAPlanner:
    def __init__(self,
                api_key: str = "EMPTY", 
                base_url: str = "http://localhost:8000/v1",
                model_name: str = None):

        transport = httpx.HTTPTransport(retries=1)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(transport=transport)
        )
        self.model = self.client.models.list().data[0].id if model_name is None else model_name
        self.log_file = None
        self.image_dir = None


    def set_logging(self, log_file, image_dir):
        self.log_file = log_file
        self.image_dir = image_dir


    def request_task(self,
            task_name: str,
            frame_path: str = None,
            instruction: str = None,
            max_token: int = 218
    ) -> str:
        if task_name == "classify_user_prompt":  # 判断是抓取具体物品还是完成一个复杂任务
            prompt = (
                f"Analyze the following user prompt: {instruction}\n\n"
                f"User prompt types:\n"
                f"- Type I (return True): User prompts with any specific descriptions\n"
                f"Examples:\n"
                f"* Color-based: \"green objects\"\n"
                f"* Position-based: \"objects from the right\"\n"
                f"* Property-based: \"all cups\"\n"
                f"* Combination: \"the red cup on the left\"\n\n"
                f"- Type II (return False): Abstract prompts without any object descriptions\n"
                f"Examples: \"clear the table\", \"clean up\", \"remove everything\"\n\n"
                f"Please determine:\n"
                f"- Is this a Type I prompt? (True/False)\n"
                f"- Provide your reasoning\n\n"
                f"Return format:\n"
                f"True/False: your reasoning\n\n"
                f"Examples:\n"
                f"- \"grab the green cup\" -> True: Contains specific object (cup) and property (green)\n"
                f"- \"clear the table\" -> False: No specific object characteristics mentioned"
            )

        elif task_name == "decompose_user_prompt":  #  抓取排序
            prompt = (
                f"For user prompt: {instruction}\n"
                f"Process:\n"
                f"1. Analyze the user prompt and image together:\n"
                f"- Match user prompt descriptions with visible objects in the image\n"
                f"- If a description (e.g., \"green objects\") matches multiple objects, include all matching objects\n"
                f"- Verify each mentioned object actually exists in the image\n\n"
                f"2. Based on the robot arm's position (right edge of the screen) and table layout\n"
                f"3. Determine the most efficient grasping sequence\n"
                f"4. Generate a reordered list of objects to grasp\n\n"
                f"Requirements:\n"
                f"- Only include objects mentioned in the original user prompt\n"
                f"- Keep position information for each object\n"
                f"- Return as a list, ordered by grasping sequence\n\n"
                f"Expected output format:\n"
                f"[\"object with position 1\", \"object with position 2\", ...]"
            )

        elif task_name == "generate_instruction":  # TODO 没有考虑instruction,直接抓取视野范围内的所有东西? 一次只给一样东西 应该是循环调用的
            prompt = (
                f"Analyze the current desktop layout and select the most suitable object to grasp, considering the following factors:\n\n"
                f"Grasping Strategy:\n"
                f"1. The robotic arm is positioned on the far right (outside the frame)\n"
                f"2. Grasping Priority Order:\n"
                f"   - Prioritize objects on the right to avoid knocking over other objects during later operations\n"
                f"   - Then consider objects in the middle\n"
                f"   - Finally, consider objects on the left\n"
                f"3. Accessibility Analysis:\n"
                f"   - Relative positions between objects\n"
                f"   - Potential obstacles\n"
                f"   - Whether the grasping path might interfere with other objects\n\n"
                f"Please provide your response in the following JSON format:\n"
                f"{{\n"
                f"    \"analysis\": {{\n"
                f"        \"priority_consideration\": \"explanation of why this object has priority\",\n"
                f"        \"accessibility\": \"analysis of object's accessibility\",\n"
                f"        \"risk_assessment\": \"potential risks in grasping this object\"\n"
                f"    }},\n"
                f"    \"target\": \"a comprehensive description of the target object (e.g., 'the blue cube on the far right of the desktop, next to the red cylinder')\"\n"
                f"}}\n\n"
                f"Ensure the output is in valid JSON format.\n"
                f"Note: The 'target' field should ONLY contain the object's color, shape, and position in a natural, flowing sentence. Do not include any analysis or reasoning in this field."
            )

        elif task_name == "mark_bounding_box":
            prompt = (
                f"Analyze the image and identify the best matching object with the description: {instruction}.\n"
                f"Instructions for object analysis:\n"
                f"1. Select ONE object that best matches the description\n"
                f"2. For the selected object, provide:\n"
                f"- A concise label, object name (3-4 words max)\n"
                f"- A detailed description (position, color, shape, context)\n"
                f"- Accurate bbox coordinates\n\n"
                f"Required JSON format with an example:\n"
                f"```json\n"
                f"{{\n"
                f"    \"bbox_2d\": [x1, y1, x2, y2],\n"
                f"    \"label\": \"green cup\",  # Keep this very brief (3-4 words)\n"
                f"    \"description\": \"a cylindrical green ceramic cup located on the right side of the wooden table, next to the laptop\"  # Detailed description\n"
                f"}}\n"
                f"```\n\n"
                f"Critical requirements:\n"
                f"- Return EXACTLY ONE object\n"
                f"- \"label\": Must be brief (3-4 words)\n"
                f"- \"description\": Must be detailed and include spatial context\n"
                f"- Use single JSON object format, not an array\n"
                f"- Ensure bbox coordinates are within image boundaries"
            )

        elif task_name == "check_grasp_success":
            prompt = (
                f"Briefly analyze the image and determine if the robotic arm has successfully grasped an object:\n"
                f"1. Consider the spatial relationship between the robotic hand and the object\n"
                f"2. Output format: explain your reasoning shortly and precisely, then conclude with a boolean value (True=grasped, False=not grasped)\n"
                f"Keep it short and simple."
            )
        
        elif task_name == "check_instruction_complete":  # TODO: check whether the prompt makes sense.
            prompt = (
                f"Please check whether {instruction} exists on the desktop. If it does not exist, output True; otherwise, output False."
            )
        
        elif task_name == "check_user_prompt_complete":
            prompt = (
                f"Please analyze the table in the image:\n\n"
                f"Requirements:\n"
                f"- Only detect physical objects with noticeable height/thickness (3D objects)\n"
                f"- Exclude from consideration:\n"
                f"* Flat items (papers, tablecloths, mats)\n"
                f"* Light projections\n"
                f"* Shadows\n"
                f"* Surface patterns or textures\n\n"
                f"Return format:\n"
                f"- True: if the table is empty of 3D objects\n"
                f"- False: if there are any 3D objects, followed by their names\n\n"
                f"Example responses:\n"
                f"True  (for empty table)\n"
                f"False: cup, bottle, plate  (for table with objects)"
            )

        else:
            raise ValueError(f"The task_name {task_name} is not a valid task name.")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        if frame_path is not None:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": frame_path}
            })
            # output the image to the image_dir
            self.save_image(frame_path, task_name)

        self.log(f"Planner requesting task: {task_name}.")
        self.log(f"Planner prompt:\n{prompt}")

        chat_completion = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_token,
            messages=messages
        )

        response = chat_completion.choices[0].message.content
        response_lower = response.lower() # 统一转为小写

        self.log(f"Planner response:\n{response}")

        if task_name == "classify_user_prompt":
            if 'true' in response_lower:
                return "TypeI"
            elif 'false' in response_lower:
                return "TypeII"
            else:
                raise ValueError(f"The output text {response} is in the wrong format.")
        elif task_name == "decompose_user_prompt":
            list_output = extract_list(response)
            if type(list_output) == list:
                return list_output
            else:
                raise ValueError(f"The output text {list_output} is not a valid list.")
        elif task_name == "generate_instruction":
            generate_task_str = parse_json(response) # 提取纯JSON
            generate_task_json = json_repair.loads(generate_task_str) # json_repair 是一个外部库，用于修复和解析可能格式不完全正确的 JSON 字符串
            generate_task = generate_task_json['target'] 
            if type(generate_task) == str:
                return generate_task
            else:
                raise ValueError(f"The output text {generate_task} is not a valid string.")
        elif task_name == "mark_bounding_box":
            bbox_str = parse_json(response)
            bbox_json = json_repair.loads(bbox_str)
            return bbox_json
        else:
            if 'true' in response_lower:
                return True
            elif 'false' in response_lower:
                return False
            else:
                raise ValueError(f"The output text {response} does not contain a valid boolean value.")

    
    def log(self, message):
        log(message, self.log_file)

    
    def save_image(self, image_url, task_name):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image = decode_base64_to_image(image_url)
        image_path = os.path.join(self.image_dir, f"{timestamp}_planner_request_{task_name}.png")
        plt.imsave(image_path, image)