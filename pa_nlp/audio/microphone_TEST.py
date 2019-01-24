#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.audio.microphone import Microphone
import optparse

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--time", type=int, default=1, help="default 5 seconds")
  parser.add_option("--export_wav_file", default="/tmp/microphone.wav",
                    help="default microphone.wav")
  (options, args) = parser.parse_args()

  microphone = Microphone()
  microphone.record(options.export_wav_file, options.time)
  microphone.play(options.export_wav_file)

if __name__  ==  '__main__':
  main()
