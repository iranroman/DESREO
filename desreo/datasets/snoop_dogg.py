# created by Iran R. Roman <iran@ccrma.stanford.edu>
import torch
import os
import numpy as np
import scipy
import librosa
from scipy.signal import butter, hilbert, filtfilt, resample

def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

class Snoop_Dogg(torch.utils.data.Dataset):

  # the dataset initialization
  def __init__(self,
               split,
               fs=22050,
               env_fs = 100,
               path_to_raw_audio = '.',
               datapoint_dur = 10,
               lowpass_cutoff = 10,
               ):
    self.audio_fs = fs
    self.env_fs = env_fs
    self.path_to_raw_audio = path_to_raw_audio
    self.split = split
    self.datapoint_dur = datapoint_dur
    self.filenames = self.get_filenames(self.path_to_raw_audio, self.split)
    self.npoints_per_file, self.ranges = self.get_song_npoints_and_idxrange()
    self.lowpass_cutoff = lowpass_cutoff

  def get_song_npoints_and_idxrange(self):
    npoints = []
    ranges = []
    for f in self.filenames:
      dur = librosa.get_duration(path=os.path.join(self.path_to_raw_audio, f))
      npoints_per_file = int(dur // self.datapoint_dur)
      ranges.append(range(sum(npoints), sum(npoints) + npoints_per_file))
      npoints.append(npoints_per_file)
    return npoints, ranges

  def get_filenames(self, path, split, test_song='song_4'):
    all_files = os.listdir(path)
    if split == 'train':
      data_files = [f for f in all_files if test_song not in f]
    elif split == 'val':
      data_files = [f for f in all_files if test_song in f]
    data_files.sort()
    return data_files

  def __len__(self):
    total_datapoints = 0
    for filename in self.filenames:
      dur = librosa.get_duration(path=os.path.join(self.path_to_raw_audio, filename))
      total_datapoints += int(dur // self.datapoint_dur)
    return total_datapoints
      
  def __getitem__(self, idx, return_envelope='True'):
    song_index = [i for i, ranges in enumerate(self.ranges) if idx in ranges][0]
    songname = self.filenames[song_index]
    x, sr = librosa.load(os.path.join(self.path_to_raw_audio,songname), 
                 offset = self.datapoint_dur * self.ranges[song_index].index(idx), 
                 duration = self.datapoint_dur, sr=self.audio_fs)

    if return_envelope:
        # envelope processing
        x_env = np.abs(hilbert(x))
        x_env = butter_lowpass_filter(x_env,self.lowpass_cutoff,self.audio_fs)
        x_env = resample(x_env,self.datapoint_dur * self.env_fs)
        x_env = np.diff(x_env)
        x_env = x_env - np.mean(x_env)
        x_env = x_env / np.std(x_env)
        return x_env
    else:
        return x

def snoop_dogg_loader(cfg, split):

    dataset = Snoop_Dogg(cfg, split)

    return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.npoints,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        )
