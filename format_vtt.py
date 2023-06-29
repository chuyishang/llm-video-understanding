import webvtt
import json

with open("./COIN/demo_coffee.txt") as f:
    data = f.readlines()
    data = data[0].split(",")
    print(data)

dict = {}
for id in data:
    entry = {}
    start, end, text = [], [], []
    try:
    for caption in webvtt.read('"/shared/medhini/COIN/asr_vtt/'+id+".en.vtt"):
        start.append(caption.start)
        end.append(caption.end)
        text.append(caption.text)
    dict[id] = {start, end, text}

with open("./COIN/json_outputs/demo_coffee.json") as outfile:
    json.dump(dict)