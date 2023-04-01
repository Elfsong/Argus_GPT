# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-03-27

import re
import pandas as pd
from datasets import load_dataset
from openai_agent import OpenAIAgent

class DataProcessor():
    def __init__(self, dataset_name, dataset_split, from_, to_,  prompt) -> None:
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.from_ = from_
        self.to_ = to_
        self.prompt = prompt
        self.dataset = [None]
        self.openai_agent = OpenAIAgent(20)
        
    def load(self):
        self.dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        self.dataset = self.dataset.select(range(self.from_, self.to_))
        return self.dataset
    
    def get_prompt(self):
        return self.prompt

    def get_num_instance(self):
        return self.to_ - self.from_
    
    def get_dataset_name(self):
        return self.dataset_name
    
    def process(self):
        raw_sources = self.get_sources()
        new_sources = [[{"role": "user", "content": f"{self.prompt}\n{source}"}] for source in raw_sources]
        responses = self.openai_agent.parallel_completion(model_name="gpt-3.5-turbo", messages=new_sources)
        pairs = self.get_pairs(raw_sources, responses)

        df = pd.DataFrame(pairs)
        df.to_csv(f'./data/{self.dataset_name}.csv', index=False)
        return df
    
    def get_sources(self):
        sources = [self.get_source(instance)for instance in self.dataset]
        return sources
    
    def get_pairs(self, raw_sources, responses):
        pairs = [self.get_pair(raw_pair) for raw_pair in zip(raw_sources, responses)]
        flat_pairs = [item for sublist in pairs for item in sublist]
        return flat_pairs
    
    def get_source(self, instance):
        raise NotImplementedError("Don't call the base class directly")
    
    def get_pair(self, raw_pair):
        raise NotImplementedError("Don't call the base class directly")


# Here you can add your custom DataProcessor, and you just need to override the get_source and get_pair methods
class YelpDataProcessor(DataProcessor):
    def get_source(self, instance):
        return instance["text"]

    def get_pair(self, raw_pair):
        source, response = raw_pair
        target = response.choices[0].message["content"].strip()
        return [{"source": source, "target": target}]

class YelpDataRandomProcessor(DataProcessor):
    def get_source(self, instance):
        return instance["text"]

    def get_pair(self, raw_pair):
        source, response = raw_pair
        target = response.choices[0].message["content"].strip()
        try:
            regex = r"\d.(.*)"
            matches = re.finditer(regex, target, re.MULTILINE)
            candidates = list()
            
            for matchNum, match in enumerate(matches, start=1):
                for groupNum in range(0, len(match.groups())):
                    groupNum = groupNum + 1
                    candidates += [match.group(groupNum)]
            
            target_candidates = candidates[0]
            
            return [{"source": source, "target": target_candidates}]
        except:
            return [{"source": source, "target": "NONE:FORMAT"}]


if __name__ == "__main__":
    # YelpData
    yelp_dp = YelpDataRandomProcessor("yelp_review_full", "train", 0, 500, "Keeping the core meaning of the following text, generate three similar texts.")
    yelp_dp.load()
    yelp_dp.process()