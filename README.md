# CUBFineGrainedClassification-NLMtextGen
# Deep Learning Coursework: Image Classification & Language Modeling

This repository contains the implementation for two distinct deep learning tasks developed: computer vision and natural language processing using TensorFlow and Keras.

##  Project Overview

### 1. Fine-Grained Image Classification (Computer Vision)
This section tackles the challenging task of fine-grained classification using the **CUB-200-2011** dataset (200 bird species). Two models were implemented and compared:
* **Transfer Learning Model (EfficientNetV2-S):** Leverages pre-trained ImageNet weights, fine-tuned in two stages (head only, then full network) with cosine learning rate decay and label smoothing.
* **Custom CNN Architecture:** A lightweight, bespoke CNN featuring parallel multi-scale convolutions (3x3 and 5x5) and channel attention to capture both fine details (e.g., beak shape) and broader patterns. 

### 2. Text Generation (Natural Language Processing)
This section focuses on building an autoregressive language model trained on the book *Poirot Investigates*. 
* **Transformer Decoder Model:** A small, custom GPT-style architecture consisting of 4 Transformer blocks, 4 causal self-attention heads, and positional encoding.
* **Word Embeddings:** Utilizes pre-trained 100-dimensional **GloVe** embeddings for rich semantic word representation prior to processing.

## Technologies & Frameworks
* **Frameworks:** TensorFlow 2.x, Keras
* **Techniques:** Transfer Learning, Multi-scale Convolution, Channel Attention, Causal Self-Attention, Label Smoothing
* **Data Handling:** `tf.data` API for efficient augmentation and pre-fetching

## 📂 How to Run
The notebook (`CUBFineGrainedClassification-NLMtextGen.ipynb`) contains clear, segmented instructions for both **Training** and **Testing**. You can download the required datasets (CUB-200-2011 via https://www.kaggle.com/datasets/xiaojiu1414/cub-200-2011 and the text corpus is available in the repository) and run the cells sequentially. Saved checkpoints can be loaded directly for evaluation without retraining.
