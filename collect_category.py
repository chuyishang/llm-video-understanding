import openai
import json
from tqdm import tqdm
import webvtt

f = open("/shared/medhini/COIN/COIN.json")
data = json.load(f)["database"]
category = []

for item in data:
    print(item)
    if data[item]["class"] == "MakeBurger":
        category.append(item)

item_list = ",".join(category)



with open("./COIN/burger/burger_list.txt","w") as f:
    f.write(item_list)