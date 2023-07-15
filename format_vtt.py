import webvtt
import json
import time

start_time = time.time()
'''
with open("./COIN/demo_coffee.txt") as f:
    data = f.readlines()
    data = data[0].split(",")
    print(data)
'''
with open("./COIN/idd_to_vtt.json") as f:
    content = json.load(f)
    data = list(content.keys())
    print(len(data))

with open("./coin_ids.txt", "w") as outfile:
    for k in data:
        outfile.write(k + ",")

f = open("./COIN/idd_to_vtt.json")
linker = json.load(f)
f.close()

output = {}
valid = []
for id in data:
    try:
        start, end, text = [], [], []
        for caption in webvtt.read(linker[id]):
            start.append(caption.start)
            end.append(caption.end)
            text.append(caption.text)
        output[id] = {"start": start, "end": end, "text": text}
        #print(linker[id])
        valid.append(id)
    except:
        print("ERROR!", id)
##print(len(output))
#print("VALID", valid)

with open("./coin_formatted_all.json", "w") as outfile:
    json.dump(output, outfile)

print(time.time() - start_time, "seconds")