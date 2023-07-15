import argparse
import json
from moviepy.editor import *


'''
with open(file_path) as f:
    for line in f:
        j_content = json.loads(line)
'''

def process_entry(entry, transcripts_path):
    #f = open(processed_path)
    #entry = json.load(f)
    video_id = entry["video_id"]
    steps = entry["steps"]
    align = entry["segments"]
    g = open(transcripts_path)
    transcripts = json.load(g)
    dict = transcripts[video_id]
    print("============")
    print(entry)
    print(dict)
    print("============")
    out = {}
    out["video_id"] = video_id
    print(len(dict["start"]))
    print("ALIGN", align)
    print("STEPS", steps)
    for num in range(1, len(steps)+1):
        step_dict = {}
        step_dict["step_label"] = steps[num-1]
        indices = align[str(num)]
        step_dict["time_start"] = dict["start"][indices[0]]
        step_dict["time_end"] = dict["end"][indices[2]]
        out[num] = step_dict
    g.close() 
    return out


def get_seconds(tstamp):
    hr = int(tstamp[0:2])
    min = int(tstamp[3:5])
    sec = float(tstamp[6:])
    return hr * 360 + min * 60 + sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path")
    parser.add_argument("--transcripts_path")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--no_formatting", action="store_true")
    parser.add_argument("--video_path")
    parser.add_argument("--output_path")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    args = parser.parse_args()

    start_idx = 0
    with open(args.processed_path) as pathfile:
        for line in pathfile:
            if start_idx < args.start:
                start_idx += 1
                continue
            entry = json.loads(line)
            output_dict = process_entry(entry, args.transcripts_path)
            print(output_dict)
            path = f"{args.output_path}{entry['video_id']}/"
            if not os.path.exists(path):
                os.mkdir(path)
            with open(path + "outfile.json", "w") as outfile:
                json.dump(output_dict, outfile)
            print(output_dict)
            for label in output_dict:
                if label == "video_id":
                    continue
                start = get_seconds(output_dict[label]["time_start"])
                end = get_seconds(output_dict[label]["time_end"])
                # DEBUG
                #print(start,end)
                if start == end or abs(start-end) < 2:
                    end = start + 1
                    start = start - 1
                elif start > end:
                    start, end = end, start
                print(start, end)
                video_file_path = args.video_path + "/" + output_dict["video_id"] + ".mp4"
                clip = VideoFileClip(video_file_path).subclip(start, end)
                try:
                    clip.write_videofile(path + "step" + str(label) + f"-{output_dict[label]['step_label']}" + ".mp4")
                except:
                    clip.write_videofile(path + "step" + str(label) + ".mp4")
            start_idx += 1




    