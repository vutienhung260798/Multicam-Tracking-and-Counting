# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1387, 877)
        font = QtGui.QFont()
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 10, 1001, 691))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setUnderline(False)
        font.setKerning(True)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setMidLineWidth(0)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(1190, 30, 89, 25))
        self.start.setObjectName("start")
        self.end = QtWidgets.QPushButton(self.centralwidget)
        self.end.setGeometry(QtCore.QRect(1190, 110, 89, 25))
        self.end.setObjectName("end")
        self.run_sever = QtWidgets.QPushButton(self.centralwidget)
        self.run_sever.setGeometry(QtCore.QRect(890, 740, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.run_sever.setFont(font)
        self.run_sever.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.run_sever.setObjectName("run_sever")
        self.connect = QtWidgets.QPushButton(self.centralwidget)
        self.connect.setGeometry(QtCore.QRect(700, 740, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.connect.setFont(font)
        self.connect.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.connect.setObjectName("connect")
        self.IP_address = QtWidgets.QLineEdit(self.centralwidget)
        self.IP_address.setGeometry(QtCore.QRect(190, 710, 481, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.IP_address.setFont(font)
        self.IP_address.setObjectName("IP_address")
        self.server_ip = QtWidgets.QLineEdit(self.centralwidget)
        self.server_ip.setGeometry(QtCore.QRect(190, 770, 481, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.server_ip.setFont(font)
        self.server_ip.setObjectName("server_ip")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 720, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(40, 780, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1387, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.start.setText(_translate("MainWindow", "start"))
        self.end.setText(_translate("MainWindow", "end"))
        self.run_sever.setText(_translate("MainWindow", "Run Sever"))
        self.connect.setText(_translate("MainWindow", "connect"))
        self.label_2.setText(_translate("MainWindow", "IP camera"))
        self.label_3.setText(_translate("MainWindow", "server ip"))
