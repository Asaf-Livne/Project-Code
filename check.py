import sys
import pickle
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QRadioButton, QButtonGroup, QFileDialog, QMessageBox, QTextEdit
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
from imports import *
from audio_data_loading import AudioDataSet as adl
import train
import test
from wavenet import WaveNetModel
import experiment

main_layout = QVBoxLayout()
# Load the uploaded images
guitar_pixmap = QPixmap("/mnt/data/jono graphics.jpg")
action_pixmap = QPixmap("/mnt/data/Action icon.jpg")
fx_pixmap = QPixmap("/mnt/data/FX icon.jpg")
load_audio_pixmap = QPixmap("/mnt/data/Load Audio icon.jpg")

# Create a label for the guitar image and title
guitar_label = QLabel()
guitar_label.setPixmap(guitar_pixmap)
guitar_label.setAlignment(Qt.AlignCenter)

title_label = QLabel("Effective ML")
title_label.setFont(QFont('Lucida Sans', 100, QFont.StyleItalic))
title_label.setStyleSheet("color: rgb(232, 230, 179);")
title_label.setAlignment(Qt.AlignCenter)

# Add guitar image and title to the layout
main_layout.addWidget(title_label)
main_layout.addWidget(guitar_label)

# Convert QPixmap to QIcon
action_icon = QIcon(action_pixmap)
fx_icon = QIcon(fx_pixmap)
load_audio_icon = QIcon(load_audio_pixmap)

# Create buttons with images
action_button = QPushButton("Action")
action_button.setIcon(action_icon)
action_button.setIconSize(action_pixmap.size())
action_button.setFixedHeight(150)
action_button.setStyleSheet("background-color: rgb(39, 37, 50); font-family:'Lucida Sans'; font-size:22px; color: rgb(232, 230, 179);")

fx_button = QPushButton("Fx")
fx_button.setIcon(fx_icon)
fx_button.setIconSize(fx_pixmap.size())
fx_button.setFixedHeight(150)
fx_button.setStyleSheet("background-color: rgb(39, 37, 50); font-family:'Lucida Sans'; font-size:22px; color: rgb(232, 230, 179);")

load_audio_button = QPushButton("Load Audio (.wav)")
load_audio_button.setIcon(load_audio_icon)
load_audio_button.setIconSize(load_audio_pixmap.size())
load_audio_button.setFixedHeight(150)
load_audio_button.setStyleSheet("background-color: rgb(39, 37, 50); font-family:'Lucida Sans'; font-size:22px; color: rgb(232, 230, 179);")

# Create a horizontal layout for the buttons
button_layout = QHBoxLayout()
button_layout.addWidget(load_audio_button)
button_layout.addWidget(fx_button)
button_layout.addWidget(action_button)

# Add buttons to the main layout
main_layout.addLayout(button_layout)

