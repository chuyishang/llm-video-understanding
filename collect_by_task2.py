import openai
import json
from tqdm import tqdm
import webvtt

f = open("/home/shang/openai-apikey.txt")
#print(f.readlines()[0])
openai.api_key = f.readlines()[0]
f.close()

f = open("./COIN/burger/burger_list.txt")
content = f.readline().split(",")

content = content[:-1]

steps = []
prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
for i in tqdm(range(len(content))):
    try:
        f = open("/shared/medhini/COIN/coin_asr/" + content[i] + ".txt")
        transcript = " ".join(f.readlines())
        #print(transcript)
        input_text = prompt.replace("|||1", transcript)
        try:
            response = openai.Completion.create(
                                    engine="text-babbage-001",
                                    prompt=input_text,
                                    temperature=0.7,
                                    max_tokens=256,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0
                                )
        except:
            print("OPENAI ERROR:", content[i])
            continue
        output = response["choices"][0]["text"].strip()
        steps.append(output)
    except:
        print(content[i])
        pass

prompt2 = "You take on the role of a professional summarizer. You are given a list of different methods to make burgers. For each method, you are given a list of steps. Use the given steps to construct a generalized recipe for making content. Do not rely on one method too much - generalize across all different methods.\nSteps: |||1\nSteps:\n1."
input_text2 = prompt2.replace("|||1", "\nMethod: ".join(steps))
response = openai.Completion.create(
                            engine="text-babbage-001",
                            prompt=input_text2,
                            temperature=0.7,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
output = response["choices"][0]["text"].strip()

with open("./COIN/burger/burger_vtt.json","w") as f:
    json.dump(output, f)