import sys
import torch
from imports import *
from audio_data_loading import AudioDataSet as adl
import train
import test
import re
from wavenet import WaveNetModel
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QRadioButton
from PyQt5.QtWidgets import QButtonGroup, QFileDialog, QMessageBox, QTextEdit, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

class EffectiveMLApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.action_result = None
        self.audio_clean_file = None
        self.audio_fx_file = None
        self.player = QMediaPlayer()  # Add a QMediaPlayer instance for audio playback

    def initUI(self):
        self.setStyleSheet("background-color: #E8E6B3; color: black;")
        # Set up the main layout
        global main_layout
        main_layout = QVBoxLayout()

        # Load images (replace with your actual paths)
        guitar_pixmap = QPixmap("images/electric guitar image.jpg")
        print(f"guitar_pixmap_size={guitar_pixmap.size()}")
        guitar_pixmap = guitar_pixmap.scaled(453*1.3, 579*1.3)  # Adjust the size (width, height) as needed
        action_pixmap = QPixmap("images/Load Audio icon.jpg")
        load_audio_pixmap = QPixmap("images/Action icon.jpg")
        load_fx_audio_pixmap = QPixmap("images/FX icon.jpg")
        load_model_pixmap = QPixmap("images/load_model.png") #need to be changed

        action_icon = QIcon(action_pixmap)
        load_audio_icon = QIcon(load_audio_pixmap)
        load_fx_audio_icon = QIcon(load_fx_audio_pixmap)
        load_model_icon = QIcon(load_model_pixmap)     

        # Create a label for the guitar image
        guitar_label = QLabel()
        guitar_label.setPixmap(guitar_pixmap)
        guitar_label.setAlignment(Qt.AlignLeft)

        # Create a label for the title
        title_label = QLabel("Effective ML")
        title_label.setFont(QFont('Lucida Sans', 60, QFont.Bold, italic=True))
        title_label.setMaximumSize(1200, 229)
        title_label.setMinimumSize(1200, 229)
        title_label.setStyleSheet("color: #FFF5DB; background-color:#272532")
        title_label.setAlignment(Qt.AlignCenter)


    # Create a horizontal layout to hold the guitar label and the title label
        header_layout = QHBoxLayout()
        header_layout.setSpacing(0)
        guitar_layout = QVBoxLayout()
        guitar_layout.addWidget(guitar_label, alignment=Qt.AlignLeft)  # Align image to the left
        right_layout = QVBoxLayout()
        right_layout.addWidget(title_label, alignment=Qt.AlignRight)  # Align title to the right
  
        
        # Convert QPixmap to QIcon
        action_icon = QIcon(action_pixmap)
        load_audio_icon = QIcon(load_audio_pixmap)
        load_fx_audio_icon = QIcon(load_fx_audio_pixmap)
        load_model_icon = QIcon(load_model_pixmap)

        load_audio_button_layout = QVBoxLayout()
        load_fx_audio_button_layout = QVBoxLayout()
        load_model_button_layout = QVBoxLayout()

        # Add a text area to display status
        self.status_text = QTextEdit()
        self.load_audio_text = QTextEdit()
        self.load_fx_audio_text = QTextEdit()
        self.load_model_text = QTextEdit()

        self.load_audio_text.setReadOnly(True)
        self.load_fx_audio_text.setReadOnly(True)
        self.load_model_text.setReadOnly(True)
        self.status_text.setReadOnly(True)

        self.load_audio_text.setStyleSheet("font-family:Lucida Sans; font-size:14px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px;border:none;")
        self.load_audio_text.setFixedHeight(30)
        #self.load_audio_text.setFixedWidth(300)

        self.load_fx_audio_text.setStyleSheet("font-family:Lucida Sans; font-size:14px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px;border:none;")
        self.load_fx_audio_text.setFixedHeight(30)
        #self.load_fx_audio_text.setFixedWidth(300)

        self.load_model_text.setStyleSheet("font-family:Lucida Sans; font-size:14px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px;border:none;")
        self.load_model_text.setFixedHeight(30)
        #self.load_model_text.setFixedWidth(200)

        # Create buttons with images
        load_audio_button = QPushButton("Load Clean Audio (.wav)")
        load_audio_button.setIcon(load_audio_icon)
        load_audio_button.setIconSize(load_audio_pixmap.size())
        load_audio_button.setFixedHeight(100)
        load_audio_button.setStyleSheet("font-family:Lucida Sans; font-size:22px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px; border:none;")
        load_audio_button.clicked.connect(self.on_clean_load_audio)

        load_fx_audio_button = QPushButton("Load Fx Audio (.wav)     ")
        load_fx_audio_button.setIcon(load_fx_audio_icon)
        load_fx_audio_button.setIconSize(load_audio_pixmap.size())
        load_fx_audio_button.setFixedHeight(100)
        load_fx_audio_button.setStyleSheet("font-family:Lucida Sans; font-size:22px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px; border:none;")
        load_fx_audio_button.clicked.connect(self.on_fx_load_audio)

        load_model_button = QPushButton("Load Model                   ")
        load_model_button.setIcon(load_model_icon)
        load_model_button.setIconSize(load_audio_pixmap.size())
        load_model_button.setFixedHeight(100)
        load_model_button.setStyleSheet("font-family:Lucida Sans; font-size:22px; color: #272532; font-style: italic; text-align: left;padding-left: 30 px;border:none;")
        load_model_button.clicked.connect(self.on_load_model)

        action_button = QPushButton("Action")
        action_button.setIcon(action_icon)
        action_button.setIconSize(load_audio_pixmap.size())
        action_button.setFixedHeight(100)
        action_button.setStyleSheet("font-family:Lucida Sans; font-size:22px; color: #272532; font-style: italic; text-align: left; padding-left: 30 px;border:none;")
        action_button.clicked.connect(self.on_action)

        # Create a horizontal layout for the buttons
        load_audio_button_layout.setSpacing(0)
        load_audio_button_layout.addWidget(load_audio_button)
        load_audio_button_layout.addWidget(self.load_audio_text)

        load_fx_audio_button_layout.setSpacing(0)
        load_fx_audio_button_layout.addWidget(load_fx_audio_button)
        load_fx_audio_button_layout.addWidget(self.load_fx_audio_text)
        
        load_model_button_layout.setSpacing(0)
        load_model_button_layout.addWidget(load_model_button)
        load_model_button_layout.addWidget(self.load_model_text)
       

        button_layout = QVBoxLayout()
        button_layout.setSpacing(0)
        button_layout.addLayout(load_audio_button_layout)
        button_layout.addLayout(load_fx_audio_button_layout)
        button_layout.addLayout(load_model_button_layout)
        button_layout.addWidget(action_button)

        right_layout.addLayout(button_layout)
        header_layout.addLayout(guitar_layout)
        header_layout.addLayout(right_layout)
        header_layout.setAlignment(right_layout, Qt.AlignTop)  # Align the right_layout to the top
        # Add buttons to the main layout
        main_layout.addLayout(header_layout)

       

        print(f"status_log: {self.status_text}")
        #main_layout.addWidget(self.status_text)

        # Set the main layout and window properties
        self.setLayout(main_layout)
        self.setWindowTitle("Effective ML")
        self.setWindowIcon(QIcon('app_icon.png'))  # Replace with your icon path
        self.showFullScreen()  # Set the window to fullscreen mode by default

    def on_clean_load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.audio_clean_file = file_name
            clean_file_name_only = file_name.split("/")
            self.load_audio_text.append(f"Loaded Clean audio: {clean_file_name_only[-1]}")

    def on_fx_load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.audio_fx_file = file_name
            fx_file_name_only = file_name.split("/")
            self.load_fx_audio_text.append(f"Loaded fx audio: {fx_file_name_only[-1]}")

    def on_load_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")
        dialog.setStyleSheet("background-color:#272532;")

        layout = QVBoxLayout()

        # Define model paths
        model_paths = {
            "EQ - Paired": "./best_models/EQ.pt",
            "Compression - Paired": "./best_models/Compression.pt",
            "Mild Distortion - Paired": "./best_models/Mild Distortion.pt",
            "Heavy Distortion - Paired": "./best_models/Heavy Distortion- Paired.pt",
            "Heavy Distortion - Unpaired": "./best_models/Heavy Distortion- Unpaired.pt",
        }

        # Create radio buttons for each model option
        button_group = QButtonGroup(dialog)
        for model_name in model_paths.keys():
            radio_button = QRadioButton(model_name)
            radio_button.setStyleSheet("font-family:Lucida Sans; font-size:12px; color: #FFF5DB; font-style: italic; text-align: left;")
            button_group.addButton(radio_button)
            layout.addWidget(radio_button)

        def submit_model():
            global best_gen_path
            selected_button = button_group.checkedButton()
            if selected_button:
                selected_model = selected_button.text()
                best_gen_path = model_paths[selected_model]  # Get the path corresponding to the selected model
                self.load_model_text.append(f"Loading model {selected_model}")
            dialog.accept()

        submit_button = QPushButton("Submit")
        submit_button.setStyleSheet("background-color:#FFF5DB; font-family:Lucida Sans; font-size:13px; color: #272532; font-weight: bold; font-style: italic; text-align: center; padding-left: 20 px;")
        layout.addWidget(submit_button)
        submit_button.clicked.connect(submit_model)

        dialog.setLayout(layout)
        dialog.exec_()

    def on_action(self):
        self.action_result = "Test"  # Automatically set the action to "Test"
        self.run_process()  # Run the process after setting the action

    def run_process(self):
        if self.action_result is None or self.audio_clean_file is None:
            QMessageBox.warning(self, "Error", "Please select an action and load an audio file first.")
            return

        self.status_text.append(f"Starting {self.action_result.lower()} with {self.audio_clean_file} and {self.audio_fx_file} ....")

        device = torch.device("cpu")
        # Set parameters based on action
        action = self.action_result.lower()
        paired = 1 
        exp_param = 0
        exp_range = 0
        cuda = False
        dilation_repeats, dilation_depth, num_channels, kernel_size, num_epochs, lr = 2, 9, 16, 5, 100, 0.0001
        
        if action == 'test':
            #best_gen_path = 'trained_generators/gen_best_model_R2_D9_C16_K5.pt'
            self.status_text.append("Testing started...")
            try:
                test.test_gen(dilation_repeats, dilation_depth, num_channels, kernel_size, best_gen_path, self.audio_clean_file, self.audio_fx_file, 44100, device)
                self.status_text.append("Testing completed successfully!")
            except Exception as e:
                self.status_text.append(f"Error during testing: {e}")

        # After process completes, call post_process
        self.post_process()
    
    def post_process(self):
        # Create a new dialog for post-process actions
        #dialog = QDialog(self)
        #dialog.setWindowTitle("Post-Process")
        #dialog.showFullScreen()  # Open the dialog in fullscreen mode

        layout = QVBoxLayout()

        title_label = QLabel("Results")
        title_label.setFont(QFont('Lucida Sans', 40, QFont.Bold, italic=True))
        title_label.setStyleSheet("color: #272532;")
        title_label.setAlignment(Qt.AlignCenter)

        # Add guitar image and title to the layout
        layout.addWidget(title_label)

        original_pixmap = QPixmap("electric_guitar.png")
        fx_pixmap = QPixmap("guitar_fx.png")
        pros_pixmap = QPixmap("NN.png")

        # Convert QPixmap to QIcon
        org_icon = QIcon(original_pixmap)
        fx_icon = QIcon(fx_pixmap)
        pros_icon = QIcon(pros_pixmap)
        
        buttons_layout = QHBoxLayout()
        # Create spacers
        left_spacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        middle_spacer_1 = QSpacerItem(50, 0, QSizePolicy.Fixed, QSizePolicy.Minimum) 
        right_spacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        # Create buttons to play the original, fx, and processed audio files
        original_button = QPushButton("Play Clean")
        #original_button.setIcon(org_icon)
        original_button.setFixedHeight(150) 
        original_button.setFixedWidth(400) 
        original_button.setStyleSheet("""
                                font-family:Lucida Sans; 
                                font-size:22px; 
                                color: #272532;
                                font-style: italic;
                                border: 1px solid #272532;
                                border-radius: 20px;
                                """)

        fx_button = QPushButton("Play Fx")
        #fx_button.setIcon(fx_icon)
        fx_button.setFixedHeight(150) 
        fx_button.setFixedWidth(400) 
        fx_button.setStyleSheet("""
                                font-family:Lucida Sans; 
                                font-size:22px; 
                                color: #272532;
                                font-style: italic;
                                border: 1px solid #272532;
                                border-radius: 20px;
                                """)
        
        processed_button = QPushButton("Play Processed")
        #processed_button.setIcon(pros_icon)
        processed_button.setFixedHeight(150) 
        processed_button.setFixedWidth(400) 
        processed_button.setStyleSheet("""
                                font-family:Lucida Sans; 
                                font-size:22px; 
                                color: #272532;
                                font-style: italic;
                                border: 1px solid #272532;
                                border-radius: 20px;
                                """)

        # Add buttons to the layout and center align them
        buttons_layout.addItem(left_spacer)
        buttons_layout.addWidget(original_button, alignment=Qt.AlignCenter)
        buttons_layout.addItem(middle_spacer_1)
        buttons_layout.addWidget(fx_button, alignment=Qt.AlignCenter)
        buttons_layout.addItem(middle_spacer_1)
        buttons_layout.addWidget(processed_button, alignment=Qt.AlignCenter)
        buttons_layout.addItem(right_spacer)
        
        layout.addLayout(buttons_layout)

        # Define the paths for the original, fx, and processed audio files
        original_audio_path = self.audio_clean_file
        fx_audio_path = self.audio_fx_file
        processed_audio_path = "./model_results/test_predictions_epoch_1.wav"  # Replace with the actual path to the processed file

        # Connect the buttons to play the corresponding audio files
        original_button.clicked.connect(lambda: self.play_audio(original_audio_path))
        fx_button.clicked.connect(lambda: self.play_audio(fx_audio_path))
        processed_button.clicked.connect(lambda: self.play_audio(processed_audio_path))

        main_layout.addLayout(layout)

    def play_audio(self, file_path):
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        self.player.play()
        self.status_text.append(f"Playing audio: {file_path}")

# Run the PyQt5 application
app = QApplication(sys.argv)
ex = EffectiveMLApp()
sys.exit(app.exec_())