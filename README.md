# Emotion Classification using RoBERTa

This project focuses on **Step 1** of building an emotion-aware chatbot: classifying user input into emotions such as sadness, joy, love, anger, and fear. The primary objective is to fine-tune a **RoBERTa model** to accurately classify text into predefined emotion categories, laying the foundation for an emotion-aware conversational agent.

---

## **Project Overview**

Emotion classification is a crucial step in understanding user sentiment and generating empathetic responses. By leveraging RoBERTa's pre-trained transformer architecture, this project achieves high accuracy and robustness in emotion prediction.

---

## **Objective**

- Classify text into one of the following emotional categories:
  - **Sadness**
  - **Joy**
  - **Love**
  - **Anger**
  - **Fear**
- Fine-tune the RoBERTa model for text classification tasks.
- Evaluate performance using precision, recall, and F1-score.

---

## **Dataset**

### **Dataset Used**
- **Name**: Kaggle Emotion Dataset
- **Description**: Contains 69,000 samples of labeled tweets with five emotion categories.
- **Link**: [Kaggle Emotion Dataset](https://www.kaggle.com/code/abdmental01/emotions-analysis-gru-94/notebook#About-the-Dataset)

### **Sample Dataset Structure**
| **ID** | **Text**                                      | **Emotion** |
|--------|-----------------------------------------------|-------------|
| 1      | "I am feeling so happy and blessed today!"    | Joy         |
| 2      | "I can't believe this happened to me."        | Sadness     |
| 3      | "You make my heart feel full of love."        | Love        |
| 4      | "This is frustrating and infuriating."        | Anger       |
| 5      | "I am scared of what might happen next."      | Fear        |

---

## **Pipeline**

### **1. Data Preprocessing**
- **Lowercasing**: Convert all text to lowercase for uniformity.
- **Stopword Removal**: Remove common words that do not add semantic meaning.
- **Punctuation Removal**: Eliminate special characters.
- **Tokenization**: Split sentences into tokens for easier processing.

### **2. Model Fine-Tuning**
- **Base Model**: Pre-trained RoBERTa model.
- **Task-Specific Head**: A classifier head added on top of RoBERTa for emotion classification.
- **Training Data**: 80% of the dataset.
- **Validation Data**: 20% of the dataset.

### **3. Model Evaluation**
- **Metrics**:
  - **Accuracy**: Overall correctness of the model.
  - **Precision and Recall**: Measure performance for each emotion category.
  - **F1-Score**: Harmonic mean of precision and recall.

---

## **Technology Stack**

1. **Model**:
   - Pre-trained RoBERTa model from Hugging Face.
2. **Libraries**:
   - `transformers`: For fine-tuning RoBERTa.
   - `torch`: PyTorch for deep learning tasks.
   - `pandas` and `numpy`: Data manipulation.
   - `scikit-learn`: Metrics and preprocessing.
   - `matplotlib` and `seaborn`: Visualization.

---

## **Installation**

### **1. Prerequisites**
- Python (>= 3.7)

### **2. Install Required Libraries**
Run the following command:
```bash
pip install transformers torch pandas numpy scikit-learn matplotlib seaborn
