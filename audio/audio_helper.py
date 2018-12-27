#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from common import *
from mutagen.flac import *
# from mutagen.mp3 import MP3
from pydub import AudioSegment
from pydub.utils import mediainfo
# import audioread
import typing

class AudioHelper:
  AUDIO_EXTENSIONS = [
    "mp3",      # converted to wav
    "flac",     # target format
    "wav",
    "sph"       # converted to wav
  ]

  @staticmethod
  def segment_audio(flac_file: str, time_segments: list,
                    dest_folder: str)-> typing.Iterator:
    '''
    time_segments: [(12,97, 18.89), (18.43, 27.77) ...] in seconds.
    return: an iterator retuning a new segment file. If one time segment is
    invalid, then the its corresponding segment file name is None.
    '''
    assert flac_file.endswith(".flac")
    assert os.path.exists(dest_folder)

    base_name = os.path.basename(flac_file)
    audio = AudioSegment.from_file(flac_file , "flac")
    duration = len(audio)
    for file_id, (time_from, time_to) in enumerate(time_segments):
      t_from = time_from * 1000
      t_to = time_to * 1000

      if not (0 <= t_from < t_to < duration):
        print(f"WARN: {flac_file} is not complete. "
              f"Actual length: {duration / 1000} seconds, "
              f"while time segment is {time_from}-{time_to}")
        yield None
        continue

      seg_name = os.path.join(dest_folder,
                              base_name.replace(".flac", f".{file_id:04}.flac"))
      audio[t_from: t_to].export(seg_name, format="flac")
      yield seg_name

  @staticmethod
  def print_audio_info(flac_file: str):
    assert flac_file.endswith(".flac")

    print(f"======= from mediainfo('{flac_file}') =======")
    info = mediainfo("sample.flac")
    print(info)
    print("=" * 32)

    audio = AudioSegment.from_file(flac_file , "flac")
    channel_count = audio.channels    #Get channels
    print(f"channel count: {channel_count}")

    sample_width = audio.sample_width #Get sample width
    print(f"sample width: {sample_width}")

    duration_in_sec = len(audio) / 1000 #Length of audio in sec
    print(f"duration: {duration_in_sec} seconds")

    sample_rate = audio.frame_rate
    print(f"sample rate: {sample_rate}")

    bit_rate = sample_width * 8
    print(f"bit rate: {bit_rate}")
    #in bytes.
    # file_size = (sample_rate * bit_rate * channel_count * duration_in_sec) / 8
    # print(f"audio file size: {file_size} bytes")

  @staticmethod
  def get_music_length(flac_file: str)-> float:
    assert flac_file.endswith(".flac")

    try:
      audio = FLAC(flac_file)
      return audio.info.length
    except Exception as error:
      print(f"ERR: {error}")
      return -1

  @staticmethod
  def convert_to_flac(in_file: str)-> typing.Union[str, None]:
    ''' Return in_file: return flac format.
    Only convert files appearing in AUDIO_EXTENSIONS'''
    file_ext = get_file_extension(in_file)

    if file_ext == "mp3":
      out_file = AudioHelper._convert_mp3_to_wav(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    elif file_ext == "flac":
      return in_file

    elif file_ext == "wav":
      out_file = AudioHelper._convert_wav_to_flac(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    elif file_ext == "sph":
      out_file = AudioHelper._convert_sph_to_wav(in_file)
      if out_file is not None:
        return AudioHelper.convert_to_flac(out_file)
      return None

    else:
      assert False, \
        f"{in_file} extension is not in {AudioHelper.AUDIO_EXTENSIONS}"

  @staticmethod
  def _convert_wav_to_flac(full_in_name: str)-> typing.Union[str, None]:
    assert full_in_name.endswith(".wav")
    out_name = full_in_name + ".flac"
    if os.path.exists(out_name):
      return out_name

    cmd = f"sox {full_in_name} {out_name}"
    if execute_cmd(cmd) == 0:
      return out_name

    return None

  @staticmethod
  def _convert_mp3_to_wav(full_in_name: str)-> typing.Union[str, None]:
    out_file = full_in_name + ".wav"
    if os.path.exists(out_file):
      return out_file

    cmd = f"ffmpeg -i {full_in_name} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    return None

  @staticmethod
  def _convert_sph_to_wav(full_in_name: str)-> typing.Union[str, None]:
    out_file = full_in_name + ".wav"
    if os.path.exists(out_file):
      return out_file

    cmd = f"sox {full_in_name} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    cmd = f"sph2pipe -f rif {full_in_name} {out_file}"
    if execute_cmd(cmd) == 0:
      return out_file

    return None

