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

class EffectiveMLApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.action_result = None
        self.audio_clean_file = None
        self.audio_fx_file = None
        self.player = QMediaPlayer()  # Add a QMediaPlayer instance for audio playback

    def initUI(self):
        self.setStyleSheet("background-color: black; color: white;")
        # Set up the main layout
        main_layout = QVBoxLayout()

        # Load images (replace with your actual paths)
        guitar_pixmap = QPixmap("Picture1.png")
        print(f"guitar_image original size = {guitar_pixmap.size()}")
        guitar_pixmap = guitar_pixmap.scaled(1500, 300, Qt.KeepAspectRatio)  # Adjust the size (width, height) as needed
        action_pixmap = QPixmap("action.png")
        load_audio_pixmap = QPixmap("load audio.png")
        load_fx_audio_pixmap = QPixmap("Fx.png")
        load_model_pixmap = QPixmap("load_model")

        # Create a label for the guitar image and title
        guitar_label = QLabel()
        guitar_label.setPixmap(guitar_pixmap)
        guitar_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Effective ML")
        title_label.setFont(QFont('Helvetica', 100))
        title_label.setStyleSheet("color: lightyellow;")
        title_label.setAlignment(Qt.AlignCenter)

        # Add guitar image and title to the layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(guitar_label)

        # Convert QPixmap to QIcon
        action_icon = QIcon(action_pixmap)
        load_audio_icon = QIcon(load_audio_pixmap)
        load_fx_audio_icon = QIcon(load_fx_audio_pixmap)
        load_model_icon = QIcon(load_model_pixmap)        
        
        # Create buttons with images
        load_audio_button = QPushButton("Load Clean Audio (.wav)")
        load_audio_button.setIcon(load_audio_icon)
        load_audio_button.setIconSize(load_audio_pixmap.size())
        load_audio_button.setFixedHeight(150) 
        load_audio_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")
        load_audio_button.clicked.connect(self.on_clean_load_audio)

        load_fx_audio_button = QPushButton("Load Fx Audio (.wav)")
        load_fx_audio_button.setIcon(load_fx_audio_icon)
        load_fx_audio_button.setIconSize(load_audio_pixmap.size())
        load_fx_audio_button.setFixedHeight(150) 
        load_fx_audio_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")
        load_fx_audio_button.clicked.connect(self.on_fx_load_audio)


        load_model_button = QPushButton("Load Model")
        load_model_button.setIcon(load_model_icon)
        load_model_button.setIconSize(load_audio_pixmap.size())
        load_model_button.setFixedHeight(150) 
        load_model_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")
        load_model_button.clicked.connect(self.on_load_model)

        action_button = QPushButton("Action")
        action_button.setIcon(action_icon)
        action_button.setIconSize(load_audio_pixmap.size())
        action_button.setFixedHeight(150) 
        action_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")
        action_button.clicked.connect(self.on_action)


        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(load_audio_button)
        button_layout.addWidget(load_fx_audio_button)
        button_layout.addWidget(load_model_button)
        button_layout.addWidget(action_button)
        

        # Add buttons to the main layout
        main_layout.addLayout(button_layout)

        # Add a text area to display status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

        # Set the main layout and window properties
        self.setLayout(main_layout)
        self.setWindowTitle("Effective ML")
        self.setStyleSheet("background-color: black;")
        self.showFullScreen()  # Set the window to fullscreen mode by default
    

    def on_clean_load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.audio_clean_file = file_name
            self.status_text.append(f"Loaded Clean audio: {self.audio_clean_file}")
    
    def on_fx_load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.audio_fx_file = file_name
            self.status_text.append(f"Loaded fx audio: {self.audio_fx_file}")
    def on_load_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")

        layout = QVBoxLayout()

        # Define model paths
        model_paths = {
            "EQ - Paired": "/Users/asaflivne/Desktop/Final Project/trained_generators/EQ.pt",
            "Compression - Paired": "/Users/asaflivne/Desktop/Final Project/trained_generators/Compression.pt",
            "Mild Distortion - Paired": "/Users/asaflivne/Desktop/Final Project/trained_generators/Mild Distortion.pt",
            "Heavy Distortion - Paired": "/Users/asaflivne/Desktop/Final Project/trained_generators/Heavy Distortion- Paired.pt",
            "Heavy Distortion - Unpaired": "/Users/asaflivne/Desktop/Final Project/trained_generators/Heavy Distortion- Unpaired.pt",
        }

        # Create radio buttons for each model option
        button_group = QButtonGroup(dialog)
        for model_name in model_paths.keys():
            radio_button = QRadioButton(model_name)
            button_group.addButton(radio_button)
            layout.addWidget(radio_button)

        def submit_model():
            global best_gen_path
            selected_button = button_group.checkedButton()
            if selected_button:
                selected_model = selected_button.text()
                best_gen_path = model_paths[selected_model]  # Get the path corresponding to the selected model
                self.status_text.append(f"Loading model {selected_model} from{best_gen_path}")  
            dialog.accept()

        submit_button = QPushButton("Submit")
        layout.addWidget(submit_button)
        submit_button.clicked.connect(submit_model)

        dialog.setLayout(layout)
        dialog.exec_()

    def on_action(self):
        self.action_result = "Test"  # Automatically set the action to "Test"
        self.run_process()  # Run the process after setting the action

    """def on_action(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Action")

        layout = QVBoxLayout()

        button_group = QButtonGroup(dialog)
        training_button = QRadioButton("Train")
        testing_button = QRadioButton("Test")
        exp_button = QRadioButton("Exp")

        button_group.addButton(training_button)
        button_group.addButton(testing_button)
        button_group.addButton(exp_button)

        layout.addWidget(training_button)
        layout.addWidget(testing_button)
        layout.addWidget(exp_button)

        def submit_action():
            if training_button.isChecked():
                self.action_result = training_button.text()  
            elif testing_button.isChecked():
                self.action_result = testing_button.text()
            elif exp_button.isChecked():
                self.action_result = exp_button.text()

            dialog.accept()  # Close the dialog
            self.sender().setEnabled(False)  # Disable the action button after submission

        submit_button = QPushButton("Submit")
        layout.addWidget(submit_button)
        submit_button.clicked.connect(submit_action)
        dialog.setLayout(layout)
        dialog.exec_()
        if self.action_result:  # Check if an action was selected
            self.run_process()  # Run the process after the dialog is closed"""




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
        dialog = QDialog(self)
        dialog.setWindowTitle("Post-Process")
        dialog.showFullScreen()  # Open the dialog in fullscreen mode

        layout = QVBoxLayout()
        button_group = QButtonGroup(dialog)

        title_label = QLabel("Results")
        title_label.setFont(QFont('Helvetica', 100))
        title_label.setStyleSheet("color: lightyellow;")
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
        
        # Create buttons to play the original, fx, and processed audio files
        original_button = QPushButton("Play Original")
        original_button.setIcon(org_icon)
        original_button.setFixedHeight(150) 
        original_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")

        fx_button = QPushButton("Play Fx")
        fx_button.setIcon(fx_icon)
        fx_button.setFixedHeight(150) 
        fx_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")

        processed_button = QPushButton("Play Processed")
        processed_button.setIcon(pros_icon)
        processed_button.setFixedHeight(150) 
        processed_button.setStyleSheet("background-color: lightyellow; font-family:Helvetica; font-size:22px; color: black;")

        
        button_group.addButton(original_button)
        button_group.addButton(fx_button)
        button_group.addButton(processed_button)

        layout.addWidget(original_button)
        layout.addWidget(fx_button)
        layout.addWidget(processed_button)

        # Define the paths for the original, fx, and processed audio files
        original_audio_path = self.audio_clean_file
        fx_audio_path = self.audio_fx_file
        processed_audio_path = "/Users/asaflivne/Desktop/Final Project/model_results/test_predictions_epoch_1.wav"  # Replace with the actual path to the processed file

        # Connect the buttons to play the corresponding audio files
        original_button.clicked.connect(lambda: self.play_audio(original_audio_path))
        fx_button.clicked.connect(lambda: self.play_audio(fx_audio_path))
        processed_button.clicked.connect(lambda: self.play_audio(processed_audio_path))

        dialog.setLayout(layout)
        dialog.exec_()

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