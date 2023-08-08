from pytube import YouTube
import os
from pathlib import Path
import subprocess

FILENAME = 'scripts/snoop_dogg.txt'
data_path = Path('data')
original_audio_path = data_path / 'original_audio'
separated_vocals_path = data_path / 'separated_vocals'

print('Loading text file with youtube URLs\n')
with open(FILENAME) as file:
    yt_urls = [line.rstrip() for line in file]


if not original_audio_path.exists():
    original_audio_path.mkdir()

if not separated_vocals_path.exists():
    separated_vocals_path.mkdir()

for i, url in enumerate(yt_urls):
  print(f'downloading song {i}')
  yt = YouTube(url)
  video = yt.streams.filter(only_audio=True).first()
  downloaded_file = video.download()
  songname_path = Path(f'song_{i}.mp3')
  filename = songname_path.stem
  Path(downloaded_file).rename(new_songname_path / songname_path)
  print(f'isolating vocals from song {i} using DEMUCS')
  subprocess.run(['demucs', '--two-stems=vocals', str(new_songname_path)])
  subprocess.run(['mv', f'separated/htdemucs/{filename}/vocals.wav', f'data/separated_vocals/{filename}.wav'])
  subprocess.run(['rm', '-r', 'separated'])
