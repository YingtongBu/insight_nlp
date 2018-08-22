#coding: utf8
# author: Xinyi Wu (xinyi.wu5@pactera.com)
import logging, logging.handlers

class BufferingSMTPHandler(logging.handlers.BufferingHandler):
  def __init__(self, mailhost, fromaddr, toaddrs, subject, capacity):
    logging.handlers.BufferingHandler.__init__(self, capacity)
    self.mailhost = mailhost
    self.mailport = None
    self.fromaddr = fromaddr
    self.toaddrs = toaddrs
    self.subject = subject
    self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %("
                                        "message)s"))

  def flush(self):
    if len(self.buffer) > 0:
      try:
        import smtplib
        port = self.mailport
        if not port:
          port = smtplib.SMTP_PORT
        smtp = smtplib.SMTP(self.mailhost, port)
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login("pingankensho@gmail.com", "PinganKensho521")
        msg = "From: {}\r\nTo: {}\r\nSubject: {}\r\n\r\n".format(
          self.fromaddr, ''.join(self.toaddrs) + ",", self.subject)
        for record in self.buffer:
          s = self.format(record)
          print(s)
          msg = msg + s + "\r\n"
        smtp.sendmail(self.fromaddr, self.toaddrs, msg)
        smtp.quit()
      except:
        self.handleError(None)  # no particular record
      self.buffer = []

if __name__ == '__main__':
    pass