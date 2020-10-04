from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import multiprocessing
import threading

class Client(object):

	def __init__(self, ip_cam = '0'):
		if ip_cam == '0':
			self.ip_cam = int(ip_cam)
		else:
			self.ip_cam = str(ip_cam)
		self.server_ip = self.get_ip_address()
		self.sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(self.server_ip))

	def get_ip_address(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(("8.8.8.8", 80))
		return s.getsockname()[0]

	def run(self):
		vs = VideoStream(src = self.ip_cam).start()
		while True:
			frame = vs.read()
			self.sender.send_image(str(self.ip_cam), frame)


if __name__ == '__main__':
	opt = Client(ip_cam = '0')
	threading.Thread(target = opt.run).start()
	# opt.run()