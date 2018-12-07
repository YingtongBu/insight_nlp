#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from mutagen.flac import *
from Common import *
from multiprocessing import Pool
import typing

class AudioHelper:
  AUDIO_EXTENSIONS = [
    "mp3",      # converted to wav
    "flac",
    "wav",
    "sph"       # converted to wav
  ]

  @staticmethod
  def seconds_to_str(seconds: float):
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    seconds = int(seconds - hours * 3600 - minutes * 60)

    return f"{hours} h {minutes} m {seconds} s"

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
  def convert_to_flac(in_file: str):
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

