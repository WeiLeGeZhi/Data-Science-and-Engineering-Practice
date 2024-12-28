import os
import json
from datetime import datetime

import ollama
import torch

zhihu_dir = "PolarDB--X/content/知乎"
path2url_file = "path2url.json"

device = "cuda"

def qwen(prompt):
    response = ollama.generate(model='qwen2.5:7b-instruct', prompt=prompt)
    return response['response']

def categorize_content(content):
    prompt = (
        "你是一个专业的文章分类专家。请问下面这篇文章的类型是技术解析吗？如果是，请回复“技术解析”；如果不是，请回复“客户案例”：\n\n"
        "内容：\n"
        f"{content}\n\n"
        "请回复如下格式：\n"
        "{'answer': '客户案例' 或 '技术解析'}"
    )
    answer = qwen(prompt)
    
    # 解析模型返回的答案
    if "客户案例" in answer:
        return "客户案例"
    elif "技术解析" in answer:
        return "技术解析"
    else:
        return "false!"


# 读取 path2url.json
with open(path2url_file, "r", encoding="utf-8") as f:
    path2url = json.load(f)

metadata_list = []

# 遍历“知乎”目录下的所有文件
for root, dirs, files in os.walk(zhihu_dir):
    for file_name in files:
        file_path = os.path.join(root, file_name)

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # 获取文件路径对应的 URL
        relative_path = os.path.relpath(file_path, zhihu_dir)
        json_path = f"知乎/{relative_path}"
        file_url = path2url.get(json_path, "")

        # 使用大模型分类
        third_directory = categorize_content(file_content)
        # if ("客户案例" in third_directory) & ("技术解析" not in third_directory):
        #     A=A+1
        # elif ("技术解析" in third_directory) & ("客户案例" not in third_directory):
        #     B=B+1
        # elif ("技术解析" in third_directory) & ("客户案例" in third_directory):
        #     C=C+1
        #     print(third_directory)
        # else:
        #     D=D+1
        #     print(third_directory)

        # 构建字典
        metadata = {
            "first_directory": "PolarDB-X",
            "second_directory": "技术专栏",
            "third_directory": third_directory,
            "file_path": f"/mnt/Langchain/knowledge_base/PolarDB--X/content/技术专栏/{file_name}",
            "file_url": file_url,
            "upload_time": datetime.now().isoformat(),
            "upload_user": "admin",
            "file_type": "External"
        }

        metadata_list.append(metadata)

# 将结果保存为 JSON 文件
with open("metadata_output.json", "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, ensure_ascii=False, indent=4)

