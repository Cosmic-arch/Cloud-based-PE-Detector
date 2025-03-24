# Cloud-based PE Malware Detection API

This repository contains the implementation of a cloud-based system for detecting malware in Portable Executable (PE) files using deep learning techniques. The project was developed as part of the DSCI 6015: Artificial Intelligence and Cybersecurity course (Spring 2025).

## Project Overview

This project implements an end-to-end malware detection system with three main components:

1. **Machine Learning Model**: A deep neural network based on the MalConv architecture, trained on the EMBER-2017 dataset to classify PE files as malicious or benign.
2. **Cloud API**: The trained model deployed as an API endpoint on AWS SageMaker for scalable inference.
3. **Web Application**: A Streamlit-based user interface that allows users to upload PE files for analysis.

## Repository Structure

```
├── notebooks/
│   ├── Malware_Detection.ipynb      # Model training and evaluation
│   ├── Deployment.ipynb             # SageMaker deployment code
│   └── Test_Endpoint.ipynb          # Testing the deployed endpoint
├── app/
│   ├── app.py                       # Streamlit web application
│   └── requirements.txt             # Python dependencies for the app
├── deployment/
│   ├── serve.py                     # Model serving code for SageMaker
│   └── model_final_2.tar.gz         # Packaged model for deployment
├── models/
│   ├── model_epoch_5.pt             # Saved model checkpoint (epoch 5)
│   ├── model_epoch_10.pt            # Saved model checkpoint (epoch 10)
│   ├── model_epoch_15.pt            # Saved model checkpoint (epoch 15)
│   ├── model_epoch_20.pt            # Saved model checkpoint (epoch 20)
│   └── model_final.pt               # Final trained model
├── data/
│   └── README.md                    # Instructions for downloading EMBER dataset
├── requirements.txt                 # Python dependencies for the project
├── Report.pdf                       # Project report (PDF)
└── README.md                        # This file
```

## Technical Approach

### 1. Building and Training the Model

- Utilized the EMBER-2017 dataset for training and evaluation
- Implemented a neural network based on the MalConv architecture using PyTorch
- Applied data preprocessing including feature standardization and reshaping
- Achieved 91.00% accuracy, 89.94% precision, and 92.33% recall on the test dataset

Key implementation files:
- `Malware_Detection.ipynb`: Contains the model implementation, training process, and evaluation

### 2. Cloud API Deployment

- Deployed the trained model as an API endpoint on AWS SageMaker
- Implemented model serving code for handling inference requests
- Configured the endpoint for optimal performance and cost-effectiveness
- Tested the endpoint with both synthetic and real PE file features

Key implementation files:
- `Deployment.ipynb`: Contains the SageMaker deployment code
- `deployment/serve.py`: Inference code for the SageMaker endpoint

### 3. Client Application

- Developed a web-based user interface using Streamlit
- Implemented PE file feature extraction using the EMBER library
- Integrated with the SageMaker endpoint for malware detection
- Created a user-friendly results display with confidence scores and visualizations

Key implementation files:
- `app/app.py`: Streamlit web application code

## Performance Analysis

The malware detection system achieved strong performance metrics:

- **Model Performance**: 91.00% accuracy, 89.94% precision, 92.33% recall, 91.12% F1 score
- **API Endpoint Performance**: Average inference time of 0.5-1.2 seconds per request
- **End-to-End Performance**: Total processing time of 2-4 seconds for a typical PE file

## Installation and Usage

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- AWS account with SageMaker access
- EMBER dataset (see instructions in `data/README.md`)

### Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pe-malware-detection.git
cd pe-malware-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Training the model:
```bash
jupyter notebook notebooks/Malware_Detection.ipynb
```

4. Deploying to SageMaker:
```bash
jupyter notebook notebooks/Deployment.ipynb
```

5. Running the Streamlit application:
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Video Demonstration

A video demonstration of the project is available at: [https://youtu.be/your-video-link](https://youtu.be/your-video-link)

## Report

A detailed report of the project is available in the `Report.pdf` file, which includes:
- Project overview and requirements
- Technical implementation details
- Performance analysis
- Limitations and future improvements
- References

## References

1. Anderson, H. S., & Roth, P. (2018). EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models. arXiv preprint arXiv:1804.04637.
2. Raff, E., Barker, J., Sylvester, J., Brandon, R., Catanzaro, B., & Nicholas, C. (2017). Malware detection by eating a whole exe. In Workshops at the Thirty-First AAAI Conference on Artificial Intelligence.
3. Amazon Web Services. (2023). Amazon SageMaker Developer Guide: Deploy Models for Inference.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- EMBER dataset creators for providing a standardized benchmark for malware detection research
- AWS SageMaker team for the documentation and examples on model deployment
