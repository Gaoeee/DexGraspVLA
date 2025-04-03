import re


def parse_json(json_output):  # 提取纯JSON
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output


def extract_list(text_output: str) -> list:
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text_output, re.DOTALL)
    if not match:
        raise ValueError(f"No list format content found: {text_output}")

    items_pattern = r'"([^"]*)"'
    items_list = re.findall(items_pattern, match.group(1))

    if not items_list:
        items_pattern = r"'([^']*)'"
        items_list = re.findall(items_pattern, match.group(1))
        
    return items_list