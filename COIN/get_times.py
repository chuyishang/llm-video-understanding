import argparse
import json
from moviepy.editor import *

def process_entry(processed_path, transcripts_path):
    f = open(processed_path)
    entry = json.load(f)
    video_id = entry["video_id"]
    steps = entry["steps"]
    align = entry["segments"]
    g = open(transcripts_path)
    transcripts = json.load(g)
    dict = transcripts[video_id]
    out = {}
    for num in range(1, len(steps)+1):
        step_dict = {}
        step_dict["step_label"] = steps[num-1]
        indices = align[str(num)]
        step_dict["time_start"] = dict["start"][indices[0]+1]
        step_dict["time_end"] = dict["end"][indices[2]]
        out[num] = step_dict
    f.close()
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
    args = parser.parse_args()

    output_dict = process_entry(args.processed_path, args.transcripts_path)
    with open(args.output_path + "outfile.json", "w") as outfile:
        json.dump(output_dict, outfile)

    
    for label in output_dict:
        start = get_seconds(output_dict[label]["time_start"])
        end = get_seconds(output_dict[label]["time_end"])
        # DEBUG
        print(start,end)
        clip = VideoFileClip(args.video_path).subclip(start, end)
        clip.write_videofile(args.output_path + "step" + str(label) + f"-{output_dict[label]['step_label']}" + ".mp4", temp_audiofile='temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")





    