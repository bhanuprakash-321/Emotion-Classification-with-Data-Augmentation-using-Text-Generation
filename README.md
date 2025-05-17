# Emotion Classification with Data Augmentation using Text Generation

## Project Overview

This project aims to accurately classify emotions expressed in text using Natural Language Processing (NLP) techniques. It addresses a common challenge in emotion detection—**class imbalance**—by generating synthetic text samples for underrepresented emotion classes using a custom Seq2Seq model with attention. The enriched dataset improves model generalization and performance, achieving significant gains in accuracy and balanced prediction across all emotion categories.

A deployment script is also included, allowing users to input their own text and receive real-time emotion predictions.


## Dataset

The dataset used is the [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) from **kaggle**, which contains text samples labeled with six emotion categories:

- **Emotions:** `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
- **Dataset Split:**
  - Training: 16,000 samples
  - Validation: 2,000 samples
  - Test: 2,000 samples

Synthetic data was generated to balance minority emotion classes and appended to the training set for augmentation.

## Methodology

### 1. Data Preprocessing
- Cleaning: Remove punctuation, numbers, URLs, mentions; lowercase all text
- Tokenization & Padding using Keras (`Tokenizer`, `pad_sequences`)
- Encoding target labels using `LabelEncoder`

### 2. Initial Emotion Classification (Imbalanced Data)
We first trained an emotion classification model using the original (imbalanced) dataset. While the model performed reasonably well on majority classes like joy and sadness, its accuracy on minority classes such as fear, surprise, and love was significantly lower. This imbalance led to skewed predictions and limited generalization.

### 3. Data Augmentation via Text Generation
To address the class imbalance issue, we implemented a custom attention-based Seq2Seq model to generate synthetic samples for underrepresented emotion classes.
- Built separate text generation models for each minority class (excluding joy)
- Incorporated **Bahdanau Attention** to maintain contextual relevance
- Used **Top-p Sampling with temperature scaling** for diverse outputs
- Applied **grammar correction** to refine generated text using language_tool_python
The newly generated samples were added to the training dataset to improve class balance.

### 4. Retraining Emotion Classification Model (Augmented Data)
After augmenting the dataset, we retrained the same classification model architecture:
- **Model Architecture:** Embedding → Bidirectional LSTM → Dense (Softmax)
- **Loss Function:** sparse_categorical_crossentropy
- **Optimizer:** Adam
- **Regularization:** Dropout layers to reduce overfitting
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
This time, the model demonstrated improved overall performance and much more balanced predictions across all emotion classes, particularly benefiting underrepresented categories.

## Results

| Metric       | Before Augmentation | After Augmentation |
|--------------|---------------------|--------------------|
| **Validation Accuracy** | ~91%               | **95%**             |
| **Test Accuracy**       | ~90%               | **95%**             |
| **Confusion Matrix**    | Balanced prediction across all emotion classes after augmentation |

The data augmentation strategy significantly improved performance and model generalization, especially on underrepresented classes like `fear`, `surprise`, and `love`.


## Business Objective

This project enables emotion-aware analysis of short-form text such as **tweets**, which are widely used to express public opinion and personal feelings.

Platforms like **Twitter** and businesses using Twitter data can benefit by:

- **Personalizing content feeds** to match user sentiment and increase engagement.
- **Tailoring marketing campaigns** by understanding user emotional response to events, brands, or trends.
- **Monitoring sentiment trends** for crisis detection, public feedback, or mental health indicators.
- **Enhancing chatbot and recommendation systems** to respond with emotional intelligence.

By improving emotional insight from tweets, platforms can foster better user experiences and businesses can make more informed decisions.


## Deployment

A simple interactive interface is provided to test the trained model:

- Input: User enters any text
- Output: The predicted **emotion** label

You can run the deployment script locally or integrate it with a UI platform like **Streamlit**.
or you can access the app from [here]()


## Conclusion

This project demonstrates how deep learning can address class imbalance in emotion classification tasks by combining NLP and generative models. The results highlight improved accuracy and balanced emotion detection, with practical implications in content personalization and social media analytics.

## Future Work
- Integrate Transformer-based models like T5 or BART for more expressive text generation
- Deploy as a full-stack application (e.g., using Streamlit)
- Enhance evaluation with human-in-the-loop validation of generated samples



