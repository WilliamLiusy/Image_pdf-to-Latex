from openai import OpenAI
from pdf2image import convert_from_path
import io
import base64
import re
import os
import argparse
# 解析命令行参数
parser = argparse.ArgumentParser(description='Convert PDF to LaTeX using AI')
parser.add_argument('input_path', help='Path to the input PDF file')
parser.add_argument('output_path', help='Path to the output LaTeX file')
args = parser.parse_args()

# 将PDF文件按页转换为图片
pdf_path = args.input_path
images = convert_from_path(pdf_path)

# 将图片转换为PNG格式的字节流，存储在内存中
image_data_list = []
for i, img in enumerate(images):
    # 创建字节流缓冲区
    img_buffer = io.BytesIO()
    # 将图片保存为PNG格式到内存中
    img.save(img_buffer, format='JPEG')
    # 获取字节数据
    img_bytes = img_buffer.getvalue()
    image_data_list.append(img_bytes)
    
    # 如果需要base64编码（用于API调用）
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key='Your Modelscope Token', # ModelScope Token
)

latex_codes = ""

# 使用第一页图片作为示例
for image_data in image_data_list:
    first_page_base64 = base64.b64encode(image_data).decode('utf-8')

    response = client.chat.completions.create(
        model='Qwen/Qwen2.5-VL-72B-Instruct', # ModelScope Model-Id
        messages=[{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': '这张图片是一页手写的数学笔记，请你帮我翻译成latex源码。请注意里面可能会有涂改，你可以按需删除他们，也可以优化排版。但是请你忠实地翻译其中的内容。',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{first_page_base64}',
                },
            }],
        }],
        stream=True
    )

    full_response = ''
    for chunk in response:
        content = chunk.choices[0].delta.content
        # print(content, end='', flush=True)
        full_response += content
        # 提取latex代码块

    matches = re.findall(r'```latex(.*?)```', full_response, re.DOTALL)
    if not matches:
        print("Warning: No LaTeX code found in the response.")
    else:
        latex_codes += "\n\n" + matches[0].strip()


# Double check by qwen3
response = client.chat.completions.create(
    model='Qwen/Qwen3-235B-A22B-Thinking-2507', # ModelScope Model-Id
    messages=[
        {
            'role': 'user',
            'content': f'请帮我检查以下latex代码是否有语法错误：\n\n```latex\n{latex_codes}\n```。请你将修改后的代码块在开始和结束都需要用```latex包裹起来输出，但请你务必保持忠实不要更改里面的含义，仅仅做语法错误和排版上的优化。'
        }
    ],
    stream=True
)

full_response = ''
done_reasoning = False
for chunk in response:
    reasoning_chunk = chunk.choices[0].delta.reasoning_content
    answer_chunk = chunk.choices[0].delta.content
    if reasoning_chunk != '':
        print(reasoning_chunk, end='',flush=True)
    elif answer_chunk != '':
        if not done_reasoning:
            print('\n\n === Final Answer ===\n')
            done_reasoning = True
        print(answer_chunk, end='',flush=True)
        full_response += answer_chunk

latex_codes = re.findall(r'```latex(.*?)```', full_response, re.DOTALL)[0].strip()

with open(args.output_path, 'w', encoding='utf-8') as f:
    f.write(latex_codes)

# os.system(f'code "{args.output_path}"')
