# from openai import OpenAI
# import os

# def get_response():
#     client = OpenAI(
#         api_key='sk-ab6eb49be7934c4f86678574618c646a', # If you have not configured the environment variable, replace DASHSCOPE_API_KEY with your API key
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # Replace https://dashscope-intl.aliyuncs.com/compatible-mode/v1 with the base_url of the DashScope SDK
#     )
#     completion = client.chat.completions.create(
#         model="qwen-plus", # Use qwn-plus as an example. You can use other models in the model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
#         messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
#                   {'role': 'user', 'content': 'Who are you?'}]
#         )
#     print(completion.model_dump_json())

# if __name__ == '__main__':
#     get_response()

from llama_index.llms.openai_like import OpenAILike

from llama_index.core.base.llms.types import LLMMetadata
class ChatOpenAILike(OpenAILike):
    @property
    def is_chat_model(self) -> bool:
        return True

llm = ChatOpenAILike(model="qwen2.5-7b-instruct-1m", 
                 api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", 
                 api_key="sk-e46fb251c74d4c64a4c2835333e994a3",
                 is_chat_model=True,
                 )

response = llm.complete("Hello World!")
print(response)