from pytube import YouTube
import os
import subprocess

FILENAME = 'scripts/snoop_dogg.txt'
DATA_DIR = 'data'
ORIGINAL_AUDIO = 'original_audio'

print('Loading text file with youtube URLs\n')
with open(FILENAME) as file:
    yt_urls = [line.rstrip() for line in file]


for i,url in enumerate(yt_urls):
  print(f'downloading song {i}')
  yt = YouTube(url)
  video = yt.streams.filter(only_audio=True).first()
  downloaded_file = video.download()
  song_name = f'song_{i}.mp3'
  os.rename(downloaded_file, song_name)
  subprocess.run(['mv',song_name, DATA_DIR+'/'+ORIGINAL_AUDIO+'/'])
  print(f'isolating vocals from song {i} using DEMUCS')
  subprocess.run(['demucs','--two-stems=vocals',DATA_DIR+'/'+ORIGINAL_AUDIO+'/'+song_name]) 
  subprocess.run(['mv',f'separated/htdemucs/song_{i}/vocals.wav',f'data/separated_vocals/song_{i}.wav'])
  subprocess.run(['rm','-r','separated'])
