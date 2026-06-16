# Seeing-Sound
Seeing Sound project in ujm created by Liu Yi and Tian Guanglan.

This repository contains the data processing, model training, deployment, and Mixed Reality (MR) visualization components of the **Seeing Sound** system. The system captures real-time environmental audio via a 4-channel microphone array, uses a deep learning model to detect car horns and sirens, and applies the GCC-PHAT algorithm to calculate the direction and distance of the sound source. The final output is visualized in 3D through a PC-based spectrogram and a Meta Quest 3 headset.

## File Structure & Description

### 1. Data Processing

This section contains the scripts and metadata required for cleaning, augmenting, and preparing the raw audio datasets for neural network training. 

* **`UrbanSound8K.csv`**: The metadata file for the primary dataset used in this project. The full dataset can be found and downloaded here: [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html).
* **`file_rename.py`**: A utility script used to clean and standardize the naming conventions of audio files gathered from various sources, ensuring compatibility with the processing pipeline.
* **`preprocessed_2.py`**: The core data processing script. It handles loading the raw `.wav` files, resampling them, converting the audio signals into Mel-spectrograms, and applying standard normalizations required by the model.
* **`data_augmentation.py`**: Applies data augmentation by primarily mixing single audio files within each dataset fold. This process creates synthetic overlapping sound environments to increase dataset diversity and significantly improve the model's robustness in real-world, noisy scenarios.
* **`merge_total.py`**: Scripts designed to aggregate various preprocessed data batches, merging them into a unified, structured dataset ready for the training phase.

### 2. Model Training

This section contains the architecture definitions, data indexing, and training loops required to build the audio classification model.

* **`ResSiren_2.py`**: The primary deep learning script. It defines the custom neural network architecture (ResSiren), configures hyperparameters, loss functions, and executes the training and validation loops to detect vehicle horns and sirens accurately.
* **`total_mel_index.csv`**: A comprehensive index mapping file. It links the data labels to their corresponding preprocessed Mel-spectrogram files, acting as a structured guide for the DataLoader to efficiently fetch batches during the training process.

### 3. Deployment

This section bridges the gap between the trained model and the real-time execution environment on edge devices.

* **`deployment.py`**: The main deployment execution script located in the root folder. It initializes the edge-computing environment, loads the trained model weights, interfaces with the 4-channel microphone array for real-time audio capturing, and bridges the inference results to the backend server for MR visualization.

> **Important Notice (ONNX File Missing)**  
> Due to GitHub's file size limitations, the pre-trained `ressiren_final.onnx` model file could not be uploaded to this remote repository. **If you wish to reproduce or run the system locally, you must execute the `deployment.py` script first.** 

### 4. 3D Visualization

This section encompasses the backend inference engine, frontend user interfaces, and 3D digital assets that work together to deliver the real-time Mixed Reality warning system.

#### Backend & Edge Inference
* **`radar_mr_backend.py`**: The core Python backend script. It handles real-time audio streams, calls the ONNX model for audio classification, uses the GCC-PHAT algorithm for distance and direction estimation, and broadcasts alert data to front-end devices in real-time via WebSockets (default port `8765`).
* **`ressiren_final.onnx`**: The exported pre-trained ResSiren deep learning model. It is optimized for real-time edge inference to efficiently identify sirens and car horns. *(Please refer to the Deployment notice above regarding the reproduction of this file).*

#### Frontend Visualization
* **`interface.html`**: The real-time spectrogram interface. Developed with Three.js, it is used to monitor and visualize the frequency and intensity changes of captured audio on a PC.
* **`3D Visualization.html`**: The MR HUD (Mixed Reality Heads-Up Display) interface designed for the Meta Quest 3. It receives alert data from the backend via WebSockets and dynamically renders 3D warning indicators within the user's field of view.

#### 3D Assets
* **`car.glb`** & **`police_car.glb`**: The 3D model assets used for the MR visualization interface. When a horn or siren is detected, the corresponding 3D vehicle model is rendered at the calculated direction and distance, providing an intuitive visual warning.
