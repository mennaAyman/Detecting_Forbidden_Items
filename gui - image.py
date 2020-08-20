# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


import sys, os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog,QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon , QPixmap
import cv2
import numpy as np
import imutils
#######################################################################
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import imutils

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
print(label_map)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print(categories)

category_index = label_map_util.create_category_index(categories)
print(category_index)

# initialize the model
detection_graph = tf.Graph()

with detection_graph.as_default():
    # initialize the graph definition
    od_graph_def = tf.GraphDef()
    
    # load the graph from disk
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as f:
        serialized_graph = f.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
    
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes

# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The score is shown on the result image, together with the class label.
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
###############################################################


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(589, 421)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Button1 = QtWidgets.QPushButton(self.centralwidget)
        self.Button1.setGeometry(QtCore.QRect(160, 280, 131, 81))
        self.Button1.setObjectName("Button1")
        self.Button2 = QtWidgets.QPushButton(self.centralwidget)
        self.Button2.setGeometry(QtCore.QRect(310, 280, 131, 81))
        self.Button2.setObjectName("Button2")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(0, 0, 590, 230))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("uu.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 589, 20))
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
        self.Button1.setText(_translate("MainWindow", "Upload image"))
        self.Button2.setText(_translate("MainWindow", "Start detection"))
    





class Widget(QtWidgets.QWidget, Ui_MainWindow):

    def __init__(self, MainWindow):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(MainWindow)
        self.Button1.clicked.connect(self.openFile)
        self.Button2.clicked.connect(self.detection)


    def detection(self, MainWindow):
        image_path = self.fileName
        print(image_path)
        text = "Negative"
        image = cv2.imread(image_path)
        before = image.copy()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        scores2 = np.squeeze(scores)
        for score in scores2:
            if score >=0.6:
                text="Positive"
                break




        # Draw the results of the detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.6)

        if text=="Positive":
            cv2.putText(image,text,(5, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image,text,(5, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        name=image_path.split(".")[0]+"-after.jpg"
        cv2.imwrite(name,image)
        cv2.imshow("image", np.hstack([before, image]))
        #cv2.imshow("image", image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        
        
        
    def openFile(self):   
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Upload image",".", "images Files (*.png *.jpg *.mp4 *.flv *.ts *.mts *.avi *.MOV)")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Widget(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
