HAM10000 Skin Lesion Classification using Transfer Learning

A deep learning project for 7-class skin lesion classification using the HAM10000 dataset and PyTorch.

The project explores transfer learning with pretrained EfficientNet-B2 and DenseNet-121, followed by full fine-tuning, class-imbalance handling, model comparison using multiple classification metrics, and deployment of the selected model as an interactive Gradio application on Hugging Face Spaces.

⚠️ Disclaimer: This project is for educational and portfolio purposes only. It is not a medical device and has not been clinically validated. Predictions should not be used for diagnosis or treatment decisions.

📌 Project Overview

The goal of this project was to build an image classification pipeline capable of distinguishing between the 7 diagnostic categories present in the HAM10000 dataset.

Instead of training a convolutional neural network completely from scratch, I used transfer learning with pretrained ImageNet models and fine-tuned them for the skin lesion classification task.

The project covers the complete workflow:

HAM10000 Dataset
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing & Augmentation
       ↓
Class Imbalance Analysis
       ↓
Class-Weighted Loss
       ↓
Transfer Learning
       ↓
Fine-Tuning
       ↓
EfficientNet-B2 vs DenseNet-121
       ↓
Multi-Metric Evaluation
       ↓
Model Selection
       ↓
PyTorch Model Serialization
       ↓
Gradio Application
       ↓
Hugging Face ZeroGPU Deployment

🧠 What I Worked On
1. Exploratory Data Analysis

The dataset was first explored using Pandas, NumPy and visualization tools to understand:

Class distribution
Number of samples per diagnostic category
Metadata
Image dimensions and characteristics
Severe class imbalance within the dataset

The analysis showed that the dataset is heavily dominated by the melanocytic nevi (nv) class, while several other classes have considerably fewer samples.

This made class imbalance an important consideration during model training.

⚖️ Handling Class Imbalance

Because the HAM10000 dataset is highly imbalanced, optimizing only for overall accuracy could lead to poor performance on minority classes.

To address this, I:

Calculated class weights from the training distribution
Used class-weighted CrossEntropyLoss
Applied data augmentation to improve generalization
Evaluated the models using class-wise precision, recall and F1-score rather than relying only on accuracy

This was particularly important for evaluating performance on clinically important minority classes such as melanoma (mel).

🔄 Transfer Learning & Fine-Tuning

I used pretrained CNN architectures instead of training from scratch.

Models evaluated
EfficientNet-B2
DenseNet-121

Both models were initialized with pretrained ImageNet weights and their final classification heads were replaced for the 7-class HAM10000 classification task.

The models were then fine-tuned on the HAM10000 dataset for 10 epochs.

Training incorporated:

CrossEntropyLoss with class weights
AdamW optimization
Weight decay for regularization
Data augmentation
Modular PyTorch Implementation

The training code was organized into reusable modules instead of keeping the entire training pipeline inside a single notebook.

The going_modular/ directory contains reusable components for areas such as:

Dataset/data loading
Model training
Evaluation
Prediction
Metrics
Training utilities

This allowed the experimentation notebook and deployment code to remain separate from reusable training components.

📊 Model Evaluation

The models were not selected based on accuracy alone.

I evaluated them using:

Accuracy
Precision
Recall
F1-score
Macro F1
Weighted F1
Confusion matrices
Class-wise performance
Melanoma precision and recall
Parameter count

This was important because the dataset contains substantial class imbalance.

Validation Results
Metric	EfficientNet-B2	DenseNet-121
Validation Accuracy	~81.1%	~83.0%
Validation Loss	~0.592	~0.597
Macro F1	~0.77	~0.77
Melanoma Recall	~0.84	~0.67
Melanoma Precision	~0.45	~0.59
Parameters	~9.11M	~7.98M
Estimated FP32 Parameter Size	~29.4 MB	~26.6 MB
Held-Out Test Results
Model	Test Accuracy	Test Loss
EfficientNet-B2	~81.3%	~0.664
DenseNet-121	~80.3%	~0.873
🔍 Model Selection

An important part of this project was understanding that the model with the highest overall accuracy is not necessarily the best model for the task.

DenseNet-121 achieved higher validation accuracy:

83.0% vs 81.1%

However, EfficientNet-B2 demonstrated substantially higher melanoma recall:

84% vs 67%

EfficientNet-B2 also achieved better performance on the held-out test set:

81.3% vs 80.3% test accuracy

Considering the class imbalance and the importance of minority-class performance, especially melanoma recall, EfficientNet-B2 was selected as the final deployment model.

📦 Deployment

The final EfficientNet-B2 model was exported and integrated into a lightweight inference application using:

PyTorch
Gradio
Hugging Face Spaces
ZeroGPU

The application allows users to upload a dermatoscopic image and receive probability scores for all seven HAM10000 classes.

🛠️ Technologies Used
Machine Learning / Deep Learning

Python
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
Computer Vision
CNNs
Transfer Learning
EfficientNet-B2
DenseNet-121
Image augmentation

Deployment
Gradio
Hugging Face Spaces

🎯 Conclusion

This project started as an exploration of transfer learning for medical image classification and evolved into a complete deep learning pipeline.

I first performed exploratory data analysis using Pandas and identified the severe class imbalance present in HAM10000. Rather than relying only on accuracy, I addressed the imbalance using class-weighted CrossEntropyLoss and evaluated the models using multiple classification metrics.

I then experimented with two pretrained architectures, EfficientNet-B2 and DenseNet-121, and fine-tuned both models for 10 epochs. Regularization through weight decay, along with data augmentation and class-weighted training, was used to improve generalization.

The comparison demonstrated an important practical lesson: model selection should depend on the requirements of the problem rather than a single metric. Although DenseNet-121 achieved higher validation accuracy, EfficientNet-B2 provided considerably higher melanoma recall and performed better on the held-out test set. Therefore, EfficientNet-B2 was selected as the final model.

Finally, I converted the trained model into a reusable inference pipeline and deployed it using Gradio on Hugging Face Spaces with ZeroGPU, turning the experimentation work into an accessible web application.

Overall, this project gave me practical experience across the complete deep learning lifecycle:

EDA → preprocessing → imbalance handling → transfer learning → fine-tuning → evaluation → model selection → deployment.

⚠️ Limitations

This project has several important limitations:

The model was trained and evaluated only on the HAM10000 dataset.
The dataset contains significant class imbalance.
The model has not been externally validated on independent datasets.
The model has not undergone clinical validation.
Performance should not be interpreted as clinical diagnostic performance.
The deployed application is intended strictly for educational and demonstration purposes.
🔮 Future Improvements

Potential improvements include:

Lesion-level rather than image-level dataset splitting
More extensive hyperparameter tuning
Longer training with learning-rate scheduling
Test-time augmentation
Calibration of predicted probabilities
Evaluation on external skin-lesion datasets
Explainability using Grad-CAM
Experimentation with Vision Transformers
Better handling of minority classes
More extensive error analysis
📎 Dataset

This project uses the HAM10000 ("Human Against Machine with 10000 training images") skin lesion dataset.

The dataset is not included in this repository. Please obtain it separately and configure the dataset paths used by the notebook.

👨‍💻 Author

Saransh Jha

Mechatronics & Automation
NIT Patna

