import torch
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    task="image-text-to-text",
    # model="deepseek-community/deepseek-vl-1.3b-chat",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    device=0,
    dtype=torch.float16
)

try:
    code_image = Image.open("bms image.png").convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": code_image},
                {"type": "text", "text": "请分析这张来源于软件技术文档："
                                         "TDD EPAM BMS system.pdf"
                                         "的图片并详细描述其中的内容。"
                                         "请先判断图片类型，包括但不限于：代码示例，流程图，架构图，"
                                         "然后根据图片类型给出详细描述，"
                                         "如果是架构图详细描述各个组件的功能和操作人员和接入方式，"
                                         "如果是代码图详细描述逻辑与输入输出，异常情况，"
                                         "如果是流程图纤细描述各个节点的作用与输入输出。"
                 }
            ]
        }
    ]

    result = pipe(
        text=messages,
        max_new_tokens=300,
        return_full_text=False
    )

    print(result[0]["generated_text"])

except FileNotFoundError:
    print("错误：找不到图片文件，请检查路径是否正确")
except Exception as e:
    print(f"发生错误：{str(e)}")
