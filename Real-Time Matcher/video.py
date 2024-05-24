from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QUrl, QSize, QTimer
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QSizePolicy, QLabel, QFileDialog, QHBoxLayout, QMessageBox,
                             QPushButton, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
import pandas as pd

from SatelliteMap import get_satellite_map
from SatelliteMap import convert_easting_northing_to_lat_long
from TimmMobilenet import TimmMobilenet

from DenseUAV import DenseUAVDataset, Mode

from PIL import Image
import torch
import time
import cv2
import torchvision.transforms as transforms
import os

from path_config import (MODEL_PATH, TRAIN_DATASET_PATH, DENSE_UAV_ROOT,
                         TEST_DATASET_PATH_0, TEST_DATASET_PATH_1, ALTO_DATASET_PATH)

class VideoPlayer(QWidget):

    def __init__(self, parent=None, sat_map_path='map_sat1.jpg'):
        super(VideoPlayer, self).__init__(parent)

        self.sat_map_path = sat_map_path
        self.metadata = None
        self.db_images = None
        self.dataset = None
        self.breakpoint = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TimmMobilenet().to(self.device)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device))

        self.model.eval()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()

        openButton = QPushButton("Open Video")
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.setIconSize(btnSize)
        openButton.setFont(QFont("Noto Sans", 8))
        openButton.setIcon(QIcon.fromTheme("document-open", QIcon("")))
        openButton.clicked.connect(self.abrir)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.label1 = QLabel("Latitude")
        self.label2 = QLabel("Longitude")

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        videoLayout = QVBoxLayout()
        videoLayout.addWidget(videoWidget)

        self.photoLabel = QLabel()
        self.photoLabel.setAlignment(Qt.AlignCenter)
        self.photoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # pixmap = QPixmap(sat_map_path).scaled(960, 420, Qt.KeepAspectRatio)#.resize((960, 420))
        # photoLabel.setPixmap(pixmap)


        displayLayout = QHBoxLayout()
        displayLayout.addWidget(videoWidget, 1)
        displayLayout.addWidget(self.photoLabel, 1)

        labelsLayout = QHBoxLayout()
        labelsLayout.addWidget(self.label1)
        labelsLayout.addWidget(self.label2)
        labelsLayout.setAlignment(Qt.AlignCenter)

        mainLayout = QVBoxLayout(self)
        mainLayout.addLayout(labelsLayout)

        mainLayout.addLayout(displayLayout)
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(self.statusBar)

        self.predictedImageLabel = QLabel()
        self.predictedImageLabel.setAlignment(Qt.AlignCenter)
        self.predictedImageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        mainLayout.addWidget(self.predictedImageLabel)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")

        self.videoCapture = None
        self.frameTimer = QTimer()
        self.frameTimer.timeout.connect(self.processNextFrame)
        self.lastFrameTime = None

        self.paused = False

    def abrir(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose video",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            print(fileName)
            if fileName == ALTO_DATASET_PATH['video']:
                self.dataset = 'ALTO'
                self.breakpoint = 275
            elif fileName == TEST_DATASET_PATH_0['video']:
                self.dataset = 'DenseUAV0'
                self.breakpoint = 200
            elif fileName == TEST_DATASET_PATH_1['video']:
                self.dataset = 'DenseUAV1'
                self.breakpoint = 599
            elif fileName == TRAIN_DATASET_PATH['video']:
                self.dataset = 'DenseUAV_train'
                self.breakpoint = 250
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Wrong video file')
                msg.setWindowTitle("Error")
                msg.exec_()
                return

            self.prepare_database()

            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()

            self.videoCapture = cv2.VideoCapture(fileName)

            self.processNextFrame()
            self.frameTimer.start(int(1000 / cv2.CAP_PROP_FPS))


    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.paused = True
            self.frameTimer.stop()
        else:
            self.mediaPlayer.play()
            if self.paused:
                self.frameTimer.start(int(1000 / self.videoCapture.get(cv2.CAP_PROP_FPS)))
                self.paused = False

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

    def processNextFrame(self):
        if not self.paused:

            start = time.time()

            ret, frame = self.videoCapture.read()
            if ret:

                pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(pil_image)

                self.performInference(pil_image)
            else:

                self.frameTimer.stop()
                self.videoCapture.release()

    def performInference(self, pil_image):

        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)
            dist = torch.cdist(output, self.db_images)

        if self.dataset == 'ALTO':
            pred_idx = torch.argsort(dist)[0][0]
            base_ref_path = ALTO_DATASET_PATH['ref_path']
            self.photoLabel.setPixmap(QPixmap(base_ref_path + os.listdir(base_ref_path)[pred_idx]))
            df = pd.read_csv(ALTO_DATASET_PATH['geo_data'])
            easting, northing = df.at[pred_idx.item(), 'easting'], df.at[pred_idx.item(), 'northing']
            lat, long = convert_easting_northing_to_lat_long(easting,northing, 18)
            self.label1.setText(str(round(lat, 5)))
            self.label2.setText(str(round(long, 5)))

        else:
            if self.dataset == 'DenseUAV_train':
                base_ref_path = TRAIN_DATASET_PATH['ref_path']
                df = pd.read_csv(TRAIN_DATASET_PATH['geo_data'], sep=' ', header=None,
                                 names=['File', 'Latitude', 'Longitude', 'Value'])
            else:
                df = pd.read_csv(TEST_DATASET_PATH_0['geo_data'], sep=' ', header=None,
                                 names=['File', 'Latitude', 'Longitude', 'Value'])
                if self.dataset == 'DenseUAV0':
                    base_ref_path = TEST_DATASET_PATH_0['ref_path']
                elif self.dataset == 'DenseUAV1':
                    base_ref_path = TEST_DATASET_PATH_0['geo_data']

            pred_idx = torch.argsort(dist)[0][0]
            df['Latitude'] = df['Latitude'].str.replace('E', '').astype(float)
            df['Longitude'] = df['Longitude'].str.replace('N', '').astype(float)
            self.label1.setText(str(round(df['Latitude'][pred_idx.item()], 5)))
            self.label2.setText(str(round(df['Longitude'][pred_idx.item()], 5)))

            self.photoLabel.setPixmap(QPixmap(base_ref_path + os.listdir(base_ref_path)[pred_idx] + '\\' + os.listdir(base_ref_path + os.listdir(base_ref_path)[pred_idx])[0]))


    def prepare_database(self):

        db_images = None
        if self.dataset == 'ALTO':
            base_ref_path = ALTO_DATASET_PATH['ref_path']
            db_images = [Image.open(base_ref_path + path).convert('RGB') for path in os.listdir(base_ref_path)[:self.breakpoint]]

            for i in range(len(db_images)):
                db_images[i] = self.transform(db_images[i])
        else:
            db_images = []
            if self.dataset == 'DenseUAV_train':
                ds = DenseUAVDataset(DENSE_UAV_ROOT, Mode.TRAIN)
                for img in ds.ref_img_path[:self.breakpoint]:
                    db_images.append(self.transform(Image.open(img)))
            else:
                ds = DenseUAVDataset(DENSE_UAV_ROOT, Mode.VAL)

                if self.dataset == 'DenseUAV0':
                    for img in ds.ref_img_path[:self.breakpoint]:
                        db_images.append(self.transform(Image.open(img)))
                elif self.dataset == 'DenseUAV1':
                    for img in ds.ref_img_path[self.breakpoint:]:
                        db_images.append(self.transform(Image.open(img)))

        db_images = torch.stack(db_images).to(self.device)

        with torch.no_grad():
            self.db_images = self.model(db_images)

