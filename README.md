# echoverse
Welcome to the Music Genre Classifier! This AI-powered application is designed to classify music into genres using deep learning. Whether you're a music lover, DJ, or just curious, this tool offers a unique way to explore and analyze your music collection.
# Features
-AI-Powered Classification: Classifies music into genres using deep learning.
-Fast & Easy: Just upload an audio file, and the app will predict the genre within seconds.
-Visualization: Displays a pie chart showing the genre composition of the audio.
-User-Friendly Interface: Built with Streamlit for an interactive and intuitive experience.
# Usage
Launch the App: Run streamlit run app.py.
Choose an Option:
About App: Overview of the app’s purpose.
How it Works?: Detailed guide on the app's functionality.
Predict Music Genre: Allows you to upload an audio file for genre prediction.
Upload Audio File: Select an mp3 file and upload it to the app.
Play Audio: Listen to the uploaded audio within the app.
Predict Genre: Click on "Know Genre" to classify the audio file. The app will display a pie chart showing genre predictions.
# how it works
The Music Genre Classifier leverages deep learning and digital signal processing techniques to analyze audio and predict genres.

1.Audio Processing: The uploaded audio file is divided into overlapping chunks to capture various sections of the song.
2.Feature Extraction: Each audio chunk is converted to a Mel Spectrogram using torchaudio. The spectrogram is resized to a (210, 210) format to match the input dimensions expected by the model.
3.Model Prediction: The pre-trained TensorFlow model classifies each chunk into a genre. The final genre is determined based on the frequency of genre predictions across all chunks.
4.Visualization: A pie chart displays the genre distribution, with the primary genre highlighted.
