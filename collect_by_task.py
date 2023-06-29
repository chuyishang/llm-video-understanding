import openai
import json
from tqdm import tqdm

f = open("/shared/medhini/COIN/COIN_annts.json")
data = json.load(f)
coffee = []

with open("./COIN/demo_coffee.txt","a+") as f:
    for id in data.keys():
        if data[id] == "MakeCoffee":
            f.write(id + ",")


steps = []
openai.api_key = "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"
prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
for i in tqdm(range(len(coffee))):
    try:
        f = open("/shared/medhini/COIN/coin_asr/" + coffee[i] + ".txt")
        transcript = " ".join(f.readlines())
        input_text = prompt.replace("|||1", transcript)
        response = openai.Completion.create(
                                engine="text-babbage-001",
                                prompt=input_text,
                                temperature=0.7,
                                max_tokens=256,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                            )
        output = response["choices"][0]["text"].strip()
        steps.append(output)
    except:
        print(coffee[i])
        pass

prompt2 = "You take on the role of a professional summarizer. You are given a list of different methods to make coffee. For each method, you are given a list of steps. Use the given steps to construct a generalized recipe for making coffee. Do not rely on one method too much - generalize across all different methods.\nSteps: |||1\nSteps:\n1."
input_text2 = prompt2.replace("|||1", "\nMethod: ".join(steps))
openai.api_key = "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"
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

with open("./COIN/general_steps.txt","a+") as f:
    f.write(output)