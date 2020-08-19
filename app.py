from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import UI
from PyQt5 import QtCore
from PyQt5.QtCore  import pyqtSlot
from PyQt5.QtGui import QImage , QPixmap
from PyQt5.QtWidgets import QDialog , QApplication
from PyQt5.uic import loadUi
import cv2
import time
from PyQt5.QtCore import QTimer
from client import Client
from server import Server
import threading
import multiprocessing


class ExampleApp(QtWidgets.QMainWindow, UI.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.connect.clicked.connect(self.run_client)
        self.run_sever.clicked.connect(self.run_server)
        # self.start.clicked.connect(self.controlTimer)

    def run_client(self):
        ip_cam = self.IP_address.text()
        server_ip = self.server_ip.text()
        client = Client(ip_cam = str(ip_cam), server_ip = str(server_ip))
        threading.Thread(target = client.run).start()
        print('run client: ' + str(ip_cam))

    def run_server(self):
        print('server started')
        if not self.timer.isActive():
            self.server = Server()
            self.timer.start(20)
            self.run_sever.setText('Running...')
        else:
            self.timer.stop()
            self.server.stop()
            self.run_sever.setText("Run Server")

    def viewCam(self):
        # print('a')
        montages = self.server.recv_frame()
        for (i, montage) in enumerate(montages):
            montage = cv2.cvtColor(montage, cv2.COLOR_BGR2RGB)
            height, width, channel = montage.shape
            step = channel * width
            qImg = QImage(montage.data, width, height, step, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qImg))

    def controlTimer(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)
            self.start.setText("Stop")
        else:
            self.timer.stop()
            self.cap.release()
            self.start.setText("Start")

def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
