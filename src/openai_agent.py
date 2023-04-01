# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-03-27

import openai
from tqdm import tqdm
from multiprocessing.pool import Pool

openai.api_key = "[OPENAI TOKEN]"

class OpenAIAgent(object):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def completion(self, closure):
        response = openai.ChatCompletion.create(
            model = closure["model_name"],
            messages = closure["messages"]
        )
        return response
    
    def parallel_completion(self, **kwargs):
        requests = [{"model_name": kwargs["model_name"], "messages": message} for message in kwargs["messages"]]
        with Pool(processes=self.num_workers) as pool:
            responses = list(tqdm(pool.imap(self.completion, requests), total=len(requests)))
        return responses

if __name__ == "__main__":
    openai_agent = OpenAIAgent(5)
    openai_agent.parallel_completion(
        model_name="gpt-3.5-turbo", 
        messages=[
            [{"role": "user", "content": "Hello!"}],
            [{"role": "user", "content": "What are you doing?"}],
        ]
    )
