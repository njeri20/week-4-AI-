# week-4-AI-

Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

Primary Differences:

Computation Graphs:

PyTorch: Uses dynamic computation graphs (define-by-run). This means the graph is built on the fly as operations are executed. This makes debugging easier (like standard Python code) and provides more flexibility for models with dynamic architectures (e.g., RNNs with variable sequence lengths).
TensorFlow (especially TensorFlow 1.x): Traditionally used static computation graphs (define-and-run). You would define the entire computational graph first, then execute it. This allowed for more low-level optimizations and easier deployment to different environments (like mobile or edge devices). However, it made debugging more challenging.
TensorFlow 2.0+: With the introduction of Eager Execution, TensorFlow now defaults to a dynamic graph similar to PyTorch, significantly bridging this gap. It still allows for static graphs (via @tf.function) for performance and deployment.
Ease of Use & Pythonic Nature:

PyTorch: Often regarded as more "Pythonic" and intuitive for Python developers due to its object-oriented design and straightforward API. It feels more like writing regular Python code.
TensorFlow: Historically had a steeper learning curve. However, with TensorFlow 2.0 and the integration of Keras as its high-level API, it has become much more user-friendly and comparable to PyTorch.
Deployment and Production Readiness:

TensorFlow: Historically had a strong advantage in production deployment due to its comprehensive ecosystem (TensorFlow Serving, TensorFlow Lite, TensorFlow Extended - TFX) for deploying models at scale across various platforms (mobile, web, edge devices).
PyTorch: Has significantly improved its production capabilities with tools like TorchServe and ONNX export, making it increasingly viable for production environments. However, TensorFlow's ecosystem is still generally more mature for end-to-end MLOps.
Community and Adoption:

PyTorch: Has gained immense popularity in the research community and academia due to its flexibility, ease of experimentation, and straightforward debugging.
TensorFlow: Backed by Google, it enjoys wide industry adoption, especially in large-scale enterprise applications and where strong MLOps integration is required.
When to choose one over the other:

Choose PyTorch when:

Research and Rapid Prototyping: Its dynamic graph and Pythonic nature allow for quick experimentation, iteration, and debugging of new model architectures.
Academic Projects: It's very popular in universities and research labs.
Projects requiring high flexibility: If your model architecture might change frequently or require complex control flow.
Choose TensorFlow when:

Large-Scale Production Deployment: Its robust ecosystem is well-suited for deploying models in production environments, especially on mobile, web, and edge devices.
Enterprise-Level Applications: Often preferred for its established stability, comprehensive tooling, and strong backing from Google.
Distributed Training: Excellent support for training very large models on massive datasets using multiple GPUs, TPUs, or distributed clusters.
When you need a full MLOps platform: TFX provides an end-to-end solution for building, deploying, and maintaining ML systems.
Q2: Describe two use cases for Jupyter Notebooks in AI development.

Exploratory Data Analysis (EDA) and Data Preprocessing:

Jupyter Notebooks provide an interactive and iterative environment ideal for understanding raw data. Data scientists can load datasets, visualize distributions with plots (histograms, scatter plots), identify outliers, handle missing values, and perform feature engineering step-by-step. Each transformation or analysis can be done in a separate cell, allowing immediate inspection of the results. This makes it easy to experiment with different preprocessing techniques and see their impact on the data before feeding it into a model. For example, you can load a CSV, display the first few rows, run df.describe(), plot a correlation matrix, and then apply StandardScaler or OneHotEncoder, observing the changes at each stage.

Model Prototyping, Training, and Evaluation:

Jupyter Notebooks are excellent for quickly prototyping and experimenting with different AI models. You can define model architectures (e.g., a simple neural network, a decision tree), train them on a subset of data, and evaluate their performance using various metrics (accuracy, loss, precision, recall) right within the notebook. This allows for rapid iteration on hyperparameters, model layers, and training strategies. Visualizing training progress (e.g., loss curves over epochs) and displaying sample predictions directly next to their true labels helps in quickly assessing model behavior and identifying areas for improvement. This "try-and-see" approach significantly speeds up the development cycle compared to running standalone scripts.
Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

spaCy significantly enhances NLP tasks compared to basic Python string operations by providing:

Linguistic Depth and Understanding:

Basic Python string operations (like str.split(), str.find(), str.replace(), regular expressions) treat text as mere sequences of characters. They lack any inherent understanding of language structure, grammar, or meaning.
spaCy, on the other hand, processes raw text into linguistically rich Doc objects. This process involves a "pipeline" that applies various linguistic annotations: 
Intelligent Tokenization: Breaks text into meaningful units (tokens), handling punctuation, contractions ("don't"), and special cases much more accurately than simple whitespace splitting.
Part-of-Speech (POS) Tagging: Assigns grammatical categories (noun, verb, adjective, etc.) to each token.
Dependency Parsing: Establishes grammatical relationships between words (e.g., subject-verb, verb-object), revealing the sentence's structure.
Lemmatization: Reduces words to their base or dictionary form (e.g., "running" -> "run", "better" -> "good", "was" -> "be"), which is crucial for reducing vocabulary size and improving model generalization.
Named Entity Recognition (NER): Identifies and classifies "named entities" such as persons, organizations, locations, dates, and products.
These rich linguistic annotations provide a much deeper, structured understanding of the text, enabling far more sophisticated and accurate NLP applications than what's possible with basic string manipulation.
Efficiency, Accuracy, and Production-Readiness:

Basic string operations often require developers to write complex, brittle, and often inefficient regular expressions or custom parsing logic for each specific linguistic task.
spaCy is designed for speed and production use. It's implemented in Cython for performance, making it highly efficient for processing large volumes of text.

It comes with pre-trained statistical models optimized for various languages, which are already highly accurate for common NLP tasks like POS tagging, parsing, and NER. This means you get high-quality results out-of-the-box without needing to train custom models initially.
Its modular design allows for creating custom processing pipelines, adding custom components, and training custom NER models, making it adaptable to specific domain needs. In essence, spaCy provides robust, pre-built, and highly optimized tools for NLP, allowing developers to focus on higher-level application logic rather than reinventing the wheel for fundamental linguistic processing.

2. Comparative Analysis
Compare Scikit-learn and TensorFlow in terms of:

Target applications (e.g., classical ML vs. deep learning):

Scikit-learn:

Target Applications: Primarily focuses on classical machine learning algorithms. It is the go-to library for traditional ML tasks involving tabular data and smaller to medium-sized datasets.
Specific Uses: Ideal for supervised learning (classification, regression - e.g., Logistic Regression, Support Vector Machines, Decision Trees, Random Forests, Gradient Boosting), unsupervised learning (clustering - e.g., K-Means, DBSCAN; dimensionality reduction - e.g., PCA, t-SNE), model selection (cross-validation, hyperparameter tuning), and data preprocessing (scaling, encoding, imputation).
When to use: When you need quick baselines, interpretability, or when deep learning might be overkill or less effective for structured data.
TensorFlow:

Target Applications: Primarily designed for deep learning and neural networks. It excels in tasks involving complex, high-dimensional, and unstructured data.
Specific Uses: Building and training various neural network architectures (Convolutional Neural Networks for images, Recurrent Neural Networks/Transformers for text/sequences, Generative Adversarial Networks, Reinforcement Learning models). It's crucial for tasks like image recognition, natural language processing (complex models), speech recognition, and complex recommendation systems.
When to use: When classical ML models are insufficient, when dealing with large datasets and complex patterns that deep learning can capture, or when deploying models at scale.
Ease of use for beginners:

Scikit-learn:

Ease of Use: Generally considered very easy for beginners to learn and use.
Reasons: It provides a consistent, intuitive, and unified API (.fit(), .predict(), .transform()) across almost all its algorithms. This uniformity drastically reduces the learning curve. Its algorithms are typically "out-of-the-box" and require less intricate setup compared to building neural networks from scratch. It allows beginners to focus on core ML concepts and model evaluation without getting bogged down in low-level computational graph details or complex network architectures.
TensorFlow:

Ease of Use: Historically had a steeper learning curve, especially with its TensorFlow 1.x static graph paradigm. However, with TensorFlow 2.0 and Keras as its high-level API, it has become significantly more user-friendly and accessible.
Reasons: Keras simplifies building neural networks into a sequential stacking of layers. While much easier now, deep learning itself involves more complex concepts (e.g., activation functions, optimizers, backpropagation, varying layer types) than classical ML. Debugging complex neural networks can still be more challenging than debugging a simple Scikit-learn model. Beginners might find initial setup (data reshaping for CNNs, one-hot encoding labels for specific losses) slightly more involved than in Scikit-learn.
Community support:

Scikit-learn:

Community Support: Has a mature, active, and well-established community.
Resources: Excellent and comprehensive official documentation, a vast number of tutorials, blog posts, and Stack Overflow answers. It's widely used in academia and industry for classical ML, leading to a rich ecosystem of shared knowledge and solutions.
TensorFlow:

Community Support: Possesses an enormous and highly dynamic community, heavily backed by Google.
Resources: Extremely extensive official documentation, numerous Google-produced tutorials, Coursera specializations, and a massive presence on GitHub and Stack Overflow. Its rapid development and position at the forefront of AI research mean a constant influx of new features, discussions, and solutions, sometimes making it challenging to keep up with the latest best practices for beginners, but also providing cutting-edge solutions.




Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal: Preprocess the data (handle missing values, encode labels), train a decision tree classifier, evaluate using accuracy, precision, and recall.
Deliverable: Python script/Jupyter notebook with comments explaining each step.

Python

# classical_ml_iris.py (or iris_classification.ipynb)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # For visualizing the decision tree

# 1. Load the Iris Species Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) # Features
y = pd.Series(iris.target) # Target labels (already numerical: 0, 1, 2)

# Display basic info about the dataset
print("--- Iris Dataset Information ---")
print(X.head())
print(y.value_counts()) # Check distribution of classes (should be balanced: 50 for each)
print(X.info()) # Check for non-null counts

# 2. Preprocess the data
# Handle missing values: The Iris dataset is famously clean and has no missing values.
# If there were missing values, you might use:
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='mean') # or 'median', 'most_frequent'
# X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("\n--- Data Preprocessing ---")
if X.isnull().sum().sum() == 0:
    print("No missing values found in the Iris dataset.")
else:
    print("Missing values handled (if any).")

# Encode labels: The 'target' attribute of load_iris() already provides numerical labels (0, 1, 2)
# corresponding to 'setosa', 'versicolor', 'virginica'. No explicit encoding is needed here
# unless you specifically loaded string labels and wanted to convert them.
# Example if starting with string labels:
# y_str = pd.Series(iris.target_names[iris.target])
# le = LabelEncoder()
# y_encoded = le.fit_transform(y_str)
print("Labels are already numerically encoded (0, 1, 2).")
print(f"Target Names: {iris.target_names}")

# 3. Split the data into training and testing sets
# We use stratify=y to ensure that the proportion of target labels is the same in both
# training and testing sets, which is important for multi-class classification.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# 4. Train a Decision Tree Classifier
print("\n--- Model Training ---")
dt_classifier = DecisionTreeClassifier(random_state=42) # Initialize the classifier
dt_classifier.fit(X_train, y_train) # Train the model on the training data
print("Decision Tree Classifier trained successfully.")

# 5. Evaluate the model
print("\n--- Model Evaluation ---")
y_pred = dt_classifier.predict(X_test) # Make predictions on the test set

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
# For multi-class classification, specify 'average' for precision and recall.
# 'weighted' accounts for class imbalance (though Iris is balanced).
# 'macro' calculates metrics for each label and finds their unweighted mean.
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

# Optional: Visualize the Decision Tree
# This requires matplotlib. If you don't have graphviz installed, plot_tree still works.
plt.figure(figsize=(20, 15))
plot_tree(dt_classifier,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, # Color nodes to indicate majority class
          rounded=True, # Round box corners
          fontsize=10)
plt.title("Decision Tree Classifier for Iris Species", fontsize=20)
plt.show()

# Additional insights: Feature Importances
print("\n--- Feature Importances ---")
feature_importances = pd.Series(dt_classifier.feature_importances_, index=iris.feature_names)
print(feature_importances.sort_values(ascending=False))
Task 2: Deep Learning with TensorFlow/PyTorch
Dataset: MNIST Handwritten Digits
Goal: Build a CNN model, achieve >95% test accuracy, visualize predictions.
Deliverable: Code with model architecture, training loop, and evaluation.

Python

# deep_learning_mnist.py (or mnist_cnn.ipynb)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Preprocess Data
print("--- Loading and Preprocessing MNIST Data ---")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] for better neural network training
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape images to (batch_size, height, width, channels) for CNN input
# MNIST images are grayscale, so channel dimension is 1.
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# One-hot encode labels (e.g., 5 -> [0,0,0,0,0,1,0,0,0,0])
# This is required for 'categorical_crossentropy' loss.
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 2. Build CNN Model
print("\n--- Building CNN Model ---")
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # Flatten the 3D output to 1D for the Dense layers
    Flatten(),
    # Dense (fully connected) layers
    Dense(128, activation='relu'), # Hidden layer
    Dense(num_classes, activation='softmax') # Output layer (10 classes for digits 0-9)
])

# Display model summary
model.summary()

# 3. Compile Model
print("\n--- Compiling Model ---")
model.compile(optimizer='adam', # Adam is a popular choice for deep learning
              loss='categorical_crossentropy', # Appropriate for one-hot encoded labels
              metrics=['accuracy']) # Track accuracy during training

# 4. Train Model
print("\n--- Training Model ---")
history = model.fit(X_train, y_train,
                    epochs=10, # Number of passes over the entire training dataset
                    batch_size=64, # Number of samples per gradient update
                    validation_split=0.1) # Use 10% of training data for validation

# Optional: Plot training history (accuracy and loss over epochs)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Evaluate Model
print("\n--- Evaluating Model on Test Set ---")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

if test_accuracy > 0.95:
    print("Goal achieved: Test accuracy > 95%!")
else:
    print("Goal not yet achieved: Test accuracy < 95%. Consider more epochs or model tuning.")

# 6. Visualize the model’s predictions on 5 sample images
print("\n--- Visualizing Predictions ---")
predictions = model.predict(X_test)

plt.figure(figsize=(12, 5))
for i in range(5):
    idx = np.random.randint(0, len(X_test)) # Pick a random image from the test set
    image = X_test[idx]
    true_label = np.argmax(y_test[idx]) # Convert one-hot to integer label
    predicted_label = np.argmax(predictions[idx]) # Get the class with highest probability

    plt.subplot(1, 5, i + 1)
    plt.imshow(image.squeeze(), cmap='gray') # .squeeze() removes the channel dimension for display
    plt.title(f"True: {true_label}\nPred: {predicted_label}",
              color='green' if true_label == predicted_label else 'red')
    plt.axis('off')
plt.suptitle('MNIST Sample Predictions (True vs. Predicted)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()

# Save the trained model for potential deployment (Bonus Task)
# model.save('mnist_cnn_model.h5')
# print("\nModel saved as 'mnist_cnn_model.h5'")
Task 3: NLP with spaCy
Text Data: User reviews from Amazon Product Reviews.
Goal: Perform named entity recognition (NER) to extract product names and brands. Analyze sentiment (positive/negative) using a rule-based approach.
Deliverable: Code snippet and output showing extracted entities and sentiment.

Python

# nlp_spacy_reviews.py (or amazon_reviews_nlp.ipynb)

import spacy
from spacy.lang.en.stop_words import STOP_WORDS # Often useful for text processing

# 1. Load spaCy Model
# Ensure you have the model downloaded: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Downloading it now...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("Download complete and model loaded.")

# 2. Text Data: User reviews from Amazon Product Reviews
amazon_reviews = [
    "The new iPhone 15 Pro Max is incredible! The camera is amazing and the battery life is fantastic. Highly recommend Apple products.",
    "This Samsung Galaxy Watch 6 is a good smartwatch, but the battery drains too quickly. Wish it had more features.",
    "Bose QuietComfort Earbuds II offer superb noise cancellation and sound quality. A bit pricey but worth it for the audio experience.",
    "Received my new Dell XPS 15 laptop. It's fast and sleek, but the trackpad occasionally acts up. Overall satisfied with Dell.",
    "This generic charger broke after a week. Don't waste your money on cheap accessories. Very disappointed.",
    "The Kindle Paperwhite is perfect for reading. Long battery life and clear display. Amazon did a great job.",
    "My Google Pixel 8 Pro camera is outstanding! The AI features are mind-blowing. Love this phone.",
    "This Logitech MX Master 3S mouse is so comfortable and precise. A must-have for productivity. Logitech makes quality peripherals.",
    "The PlayStation 5 is an excellent gaming console with stunning graphics, but finding games can be expensive.",
    "This cheap knockoff HDMI cable constantly disconnects. Absolutely terrible quality. Avoid at all costs."
]

print("\n--- Performing Named Entity Recognition (NER) ---")
# 3. Perform Named Entity Recognition (NER) to extract product names and brands
for i, review in enumerate(amazon_reviews):
    doc = nlp(review)
    print(f"\nReview {i+1}: \"{review}\"")
    print("  Extracted Entities (Product/Brand Focus):")
    found_entities = []
    # spaCy's default NER may identify various entity types. We focus on common ones for products/brands.
    # Common labels: ORG (Organization), PRODUCT, GPE (Geo-Political Entity, sometimes used for companies)
    # Consider extending with custom rules or a custom NER model for higher precision on specific products.
    relevant_labels = ["ORG", "PRODUCT", "GPE", "FAC", "PERSON"] # PERSON sometimes for founders/designers
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            # Simple heuristic: if the entity text contains numbers (like iPhone 15) or known brand names
            # This makes the extraction more focused on what you'd consider a "product" or "brand"
            if any(char.isdigit() for char in ent.text) or \
               any(brand in ent.text for brand in ["iPhone", "Apple", "Samsung", "Bose", "Dell", "Kindle", "Amazon", "Google", "Pixel", "Logitech", "PlayStation"]):
                found_entities.append(f"  - {ent.text} ({ent.label_})")
    
    if found_entities:
        print("\n".join(found_entities))
    else:
        print("  No primary product/brand entities identified by default NER with applied heuristic.")

# 4. Analyze sentiment (positive/negative) using a rule-based approach
print("\n--- Performing Rule-Based Sentiment Analysis ---")

# Define keywords for positive and negative sentiment
# These are just examples; for a real system, you'd use a more comprehensive lexicon
positive_keywords = {"incredible", "amazing", "fantastic", "highly recommend", "superb", "worth",
                     "satisfied", "good", "fast", "sleek", "perfect", "clear", "outstanding",
                     "mind-blowing", "love", "comfortable", "precise", "must-have", "quality", "excellent", "stunning"}
negative_keywords = {"drains too quickly", "wish it had more features", "pricey", "acts up",
                     "broke", "don't waste", "disappointed", "cheap", "terrible", "avoid", "disconnects", "expensive"}

def analyze_sentiment_rule_based(text):
    doc = nlp(text.lower()) # Process the text and convert to lowercase for consistent matching
    pos_score = 0
    neg_score = 0
    
    # Simple negation handling: look for "not" before a keyword
    negation_tokens = {"not", "no", "n't", "never", "without"}

    for token in doc:
        # Check for negation in previous tokens
        is_negated = False
        if token.i > 0: # Check the token before
            if token.nbor(-1).text.lower() in negation_tokens:
                is_negated = True
            elif token.i > 1 and token.nbor(-2).text.lower() in negation_tokens: # Check two tokens before
                 is_negated = True # Simple check, can be more sophisticated

        if token.text in positive_keywords:
            if is_negated:
                neg_score += 1 # "not good" becomes negative
            else:
                pos_score += 1
        elif token.text in negative_keywords:
            if is_negated:
                pos_score += 1 # "not bad" becomes positive
            else:
                neg_score += 1
    
    # Determine overall sentiment
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral" # Or you could count for specific neutral terms

for i, review in enumerate(amazon_reviews):
    sentiment = analyze_sentiment_rule_based(review)
    print(f"\nReview {i+1}: \"{review}\"")
    print(f"  Sentiment: {sentiment}")

1. Ethical Considerations
Identify potential biases in your MNIST or Amazon Reviews model. How could tools like TensorFlow Fairness Indicators or spaCy’s rule-based systems mitigate these biases?

Let's analyze potential biases for both models and discuss mitigation:

A. MNIST Handwritten Digits Model (CNN)

Potential Biases:

Representation Bias (Data Bias): While MNIST is a classic and relatively clean dataset, it's compiled from specific sources (NIST's Special Database 3 and Special Database 1, predominantly US Census Bureau employees and high school students). This means the handwriting styles within the dataset might not be fully representative of the diversity of handwriting across different demographics (e.g., age groups, countries, cultures, education levels, physical conditions like tremors). A model trained solely on this might perform worse on digits written by individuals whose handwriting styles are underrepresented in the training data, leading to a biased or unfair system if used in a real-world application (e.g., digitizing forms for a global user base).
Annotation Bias (less likely for MNIST): For a dataset as straightforward as MNIST, annotation bias (inconsistent or subjective labeling) is minimal. However, in more complex image recognition tasks, human annotators might implicitly introduce biases based on their own perceptions or instructions.
Mitigation with TensorFlow Fairness Indicators:

TensorFlow Fairness Indicators is a suite of tools built on TensorFlow Model Analysis (TFMA) that helps in evaluating model fairness by slicing performance metrics (accuracy, precision, recall, etc.) across different sub-groups of data.
How it could mitigate bias in a more general handwriting recognition task (extrapolating from MNIST):
Define Subgroups: To effectively use Fairness Indicators, you'd need metadata associated with your handwriting samples (e.g., writer's age range, geographic region, education level, gender). If MNIST had such metadata, we could define slices like age_group: 'elderly', region: 'Asia', etc.
Slice Performance Evaluation: Fairness Indicators would then calculate and visualize metrics (e.g., prediction accuracy for each digit class) for each of these defined subgroups.
Identify Disparities: If the tool reveals significantly lower accuracy or higher error rates for certain subgroups (e.g., digits written by elderly individuals are consistently misclassified), it flags a fairness issue.
Root Cause Analysis and Action: These disparities would prompt investigation into the training data (e.g., underrepresentation of handwriting styles from specific age groups). Mitigation strategies could include:
Data Collection/Augmentation: Actively collecting more diverse handwriting samples from underrepresented groups.
Re-weighting/Re-sampling: Adjusting the importance of samples during training to give more weight to underperforming groups.
Fairness-aware Regularization: Incorporating fairness constraints into the model's loss function to explicitly reduce performance disparities.
Model Re-evaluation: After applying mitigation, re-evaluate with Fairness Indicators to confirm improvement across all slices.
B. Amazon Reviews Sentiment Model (Rule-Based)

Potential Biases:

Lexical Bias / Keyword Dependency: This is the most significant bias. Our rule-based system relies entirely on a fixed list of positive and negative keywords.
Contextual Ambiguity: Words like "sick" can be negative in health ("I feel sick") but positive in slang ("that car is sick!"). A simple keyword match will miss this nuance.
Sarcasm and Irony: "Oh, great, my new phone exploded!" will be wrongly classified as positive. Rule-based systems struggle immensely with non-literal language.
Domain Specificity: A word like "light" might be positive for a laptop ("lightweight laptop") but neutral or negative for a heavy-duty tool ("this hammer is too light"). Our generic keywords don't account for domain-specific sentiment.
Cultural/Linguistic Variation: Sentiment expressed in one dialect or cultural context might use different phrases or word intensities than assumed by our keyword list, leading to misclassification for certain user groups.
Evolution of Language: New slang terms constantly emerge; a fixed keyword list quickly becomes outdated.
Negation Handling Simplification: Our simple negation logic ("not X") is primitive. Complex negations ("hardly ever works", "rarely disappoints") or double negatives ("not unhelpful") would likely be misinterpreted.
Absence of Intensity/Emphasis: "Good" and "extremely good" get the same weight, even though the latter implies stronger positive sentiment.
Mitigation with spaCy’s Rule-Based Systems (Advanced Use):

While our initial implementation was basic, spaCy's rule-based matching capabilities (Matcher, PhraseMatcher) are much more powerful and can be used to mitigate some of these biases by capturing more linguistic complexity:
Contextual Patterns (Matcher): Instead of just checking for single words, Matcher allows defining patterns based on multiple token attributes (lemma, POS tag, dependency relation) and their sequence.
Improved Negation Handling: Create sophisticated patterns like [{"LEMMA": {"IN": ["not", "never"]}}, {"POS": {"IN": ["ADJ", "ADV"]}, "OP": "+"}, {"LEMMA": {"IN": ["good", "bad"]}}] to capture negations more reliably and reverse sentiment.
Sarcasm/Idiom Detection: While hard, you can define patterns for known sarcastic phrases or idioms (e.g., "Bless your heart" for negative sentiment in some contexts) if you have a finite list.
Domain-Specific Lexicons (PhraseMatcher): Instead of generic keywords, you can load large lists of domain-specific terms (e.g., from product review corpora) into PhraseMatcher. This helps in correctly interpreting words that have different sentiment valences in the product review domain (e.g., "buggy" for software is negative, "crisp" for display is positive).
Dependency-Based Rules: Leverage spaCy's dependency parser to understand how words relate. For example, a rule could state: if a "negative" adjective modifies a "product" noun via an "amod" dependency, then the product is likely viewed negatively. This provides much more precision than simple keyword counting.
Combining Rules with Statistical Models: The most effective mitigation often involves a hybrid approach. Rules can handle clear-cut, high-precision cases, while a statistical model (trained on diverse data) handles more nuanced, complex, or ambiguous sentiment. spaCy allows integrating custom components, so rule-based sentiment can be a pre-processing step or a component in a larger NLP pipeline that includes a machine learning classifier.
Iterative Refinement: Rule-based systems allow for direct intervention. When a bias or misclassification is identified (e.g., through user feedback or manual review), new rules can be added or existing ones modified specifically to address that pattern, providing transparent and controllable bias mitigation.


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical # FIXED: Imported to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# FIXED 1: Image data preprocessed for CNN (normalization and channel dimension)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, -1) # Add channel dimension (grayscale: 1)
X_test = np.expand_dims(X_test, -1)

# FIXED 2: Labels one-hot encoded for 'categorical_crossentropy'
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential([
    # Input shape is now correct due to Fix 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), # Added an extra Conv layer for better performance
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'), # Added a hidden Dense layer for more complexity
    Dense(num_classes, activation='softmax') # FIXED 4: Changed activation to 'softmax' for multi-class
])

# FIXED 5: Correct loss function for one-hot encoded labels and softmax output
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# FIXED 6: Increased epochs and added validation split to monitor training
history = model.fit(X_train, y_train,
                    epochs=10, # Increased epochs for better accuracy
                    batch_size=64,
                    validation_split=0.1) # Monitor validation performance

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

