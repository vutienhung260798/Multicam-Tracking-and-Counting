from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import multiprocessing
import threading

class Client(object):

	def __init__(self, ip_cam = '0', server_ip = ''):
		if ip_cam == '0':
			self.ip_cam = int(ip_cam)
		else:
			self.ip_cam = 'http://' + str(ip_cam) +'/video?.mjpeg'
		self.server_ip = server_ip
		self.sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(self.server_ip))

	def run(self):
		vs = VideoStream(src = self.ip_cam).start()
		while True:
			frame = vs.read()
			self.sender.send_image(str(self.ip_cam), frame)


if __name__ == '__main__':
	opt = Client(ip_cam = '0', server_ip = '192.168.100.111')
	threading.Thread(target = opt.run).start()
	# opt.run()