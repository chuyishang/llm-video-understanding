# Import everything needed to edit video clips
from moviepy.editor import *

def get_seconds(tstamp):
    hr = int(tstamp[0:2])
    min = int(tstamp[3:5])
    sec = float(tstamp[6:])
    #print(hr, min, sec)
    #print(hr * 360 + min * 60 + sec)
    return hr * 360 + min * 60 + sec

timestamps = {1: {'step_label': 'Get a metal pot',
  'time_start': '00:00:03.649',
  'time_end': '00:00:31.820'},
 2: {'step_label': 'Ground coffee (coarse or fine)',
  'time_start': '00:00:31.820',
  'time_end': '00:00:34.190'},
 3: {'step_label': 'Sugar (optional)',
  'time_start': '00:00:34.190',
  'time_end': '00:00:38.630'},
 4: {'step_label': 'Coffee for three people',
  'time_start': '00:00:38.630',
  'time_end': '00:01:43.999'},
 5: {'step_label': 'Add ground coffee to the pot',
  'time_start': '00:01:43.999',
  'time_end': '00:02:23.430'},
 6: {'step_label': 'Bring the pot to a boil',
  'time_start': '00:02:25.650',
  'time_end': '00:02:30.120'},
 7: {'step_label': 'Remove the pot from the heat',
  'time_start': '00:02:30.120',
  'time_end': '00:02:44.750'},
 8: {'step_label': 'Pour the coffee into the cups',
  'time_start': '00:02:47.220',
  'time_end': '00:03:39.170'},
 9: {'step_label': 'Enjoy',
  'time_start': '00:03:39.170',
  'time_end': '00:03:41.480'},
 }

#print(get_seconds(timestamps[5]["time_start"]))


for label in timestamps:
    start = get_seconds(timestamps[label]["time_start"])
    end = get_seconds(timestamps[label]["time_end"])
    clip = VideoFileClip("TvNWLrRzIAM.mp4").subclip(start, end)
    clip.write_videofile("./clips/step" + str(label) + ".mp4", temp_audiofile='temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
