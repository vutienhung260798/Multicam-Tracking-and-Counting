from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
ap.add_argument("-i", "--ip-camera", required = True, 
	help="ip address of camera")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
	args["server_ip"]))

if args['ip_camera'] == "0":
	ip = int(args['ip_camera'])
else:
	ip = 'http://' + str(args['ip_camera']) +'/video?.mjpeg'
# get the host name, initialize the video stream, and allow the
rpiName = socket.gethostname()
vs = VideoStream(src=ip).start()
 
while True:
	# read the frame from the camera and send it to the server
	frame = vs.read()
	sender.send_image(str(ip), frame)