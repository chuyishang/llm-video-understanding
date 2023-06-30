from __future__ import unicode_literals
import youtube_dl

with open("./COIN/demo_coffee.txt") as f:
    data = f.readlines()[0].split(",")

for id in data:
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v=5aDqNwMVEyc'])