## Technical Report: Spam/Ham Classification using BERT

**1. Introduction**

This report details the implementation of a spam/ham email classification system using the Bidirectional Encoder Representations from Transformers (BERT) model.  Moving beyond traditional methods like Naive Bayes, which relies on word frequency and independence assumptions, BERT leverages the power of transformer networks and pre-trained language understanding to achieve more sophisticated text classification. This report will delve into the code implementation, explaining each step from data preparation to model evaluation, with a focus on the underlying principles of BERT and its application to this specific task.

**2. Data Preprocessing**

The initial steps of data preprocessing are similar to those employed for simpler models, ensuring data quality and suitability for the chosen model.

```python
# %%
import torch
import sklearn
import pandas as pd
import csv

# %%
SpamHam = pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
SpamHam.head(5)

# %%
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

SpamHam['clean_text'] = SpamHam['text'].apply(clean_text)
SpamHam['clean_text'].head(5)
```

*   **`SpamHam = pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')`**: This line reads the dataset from a CSV file into a Pandas DataFrame.  The DataFrame `SpamHam` now holds the email data, including the text content and labels (spam or ham).

*   **`def clean_text(text): ... return text`**: This function defines a text cleaning process:
    *   **`text = text.lower()`**: Converts all text to lowercase, standardizing the input.
    *   **`text = re.sub(r'[^a-z\s]', '', text)`**: Removes any characters that are not lowercase letters or whitespace. This step eliminates punctuation and numbers, focusing on textual content.
    *   **`text = re.sub(r'\s+', ' ', text).strip()`**:  Replaces multiple whitespace characters with a single space and removes leading/trailing whitespace, normalizing spacing in the text.
*   **`SpamHam['clean_text'] = SpamHam['text'].apply(clean_text)`**: Applies the `clean_text` function to the 'text' column of the DataFrame, creating a new column 'clean\_text' containing the processed email text.

**3. BERT Model and Tokenization**

The core of this implementation is the use of BERT, a powerful pre-trained transformer model. Unlike Bag-of-Words approaches that treat words in isolation, BERT understands context and relationships between words in a sentence.

```python
# %%
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
os.environ['WANDB_DISABLED'] = 'true'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

*   **`from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments ...`**: Imports necessary components from the `transformers` library by Hugging Face, which provides pre-trained models and tools for Natural Language Processing.
    *   **`BertTokenizer`**:  Used to convert raw text into tokens that BERT can understand.
    *   **`BertForSequenceClassification`**:  A BERT model specifically designed for sequence classification tasks like spam/ham detection.
    *   **`Trainer`**: A class that simplifies the training and fine-tuning process of transformer models.
    *   **`TrainingArguments`**:  A class to configure various training parameters.
    *   **`Dataset`, `DataLoader`**: PyTorch utilities for handling datasets and creating batches for training.

*   **`tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`**: Initializes a BERT tokenizer.
    *   **Tokenization Process**: BERT uses WordPiece tokenization. This involves breaking down words into subword units if they are not in the tokenizer's vocabulary. For example, "unbelievable" might be tokenized into "un", "##believ", "##able". This helps BERT handle out-of-vocabulary words and captures sub-word meaning.
    *   **Vocabulary**: The tokenizer comes with a pre-defined vocabulary learned during BERT's pre-training. This vocabulary covers a vast range of words and subwords.
    *   **Special Tokens**: The tokenizer also handles special tokens crucial for BERT:
        *   **`[CLS]` (Classification)**: Added at the beginning of each input sequence. Its representation in the final layer is used for classification tasks.
        *   **`[SEP]` (Separator)**: Used to separate sentences in tasks like sentence pair classification (not directly relevant in this single-sentence classification, but still part of the standard BERT input format).

*   **`model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)`**: Loads a pre-trained BERT model fine-tuned for sequence classification.
    *   **Pre-trained BERT**:  `'bert-base-uncased'` specifies using the base, uncased version of BERT. This model has been pre-trained on a massive corpus of text data using two unsupervised tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). This pre-training allows BERT to learn general language representations.
    *   **`BertForSequenceClassification`**:  This class adds a linear classification layer on top of the base BERT model. This layer takes the representation of the `[CLS]` token from the final BERT layer and projects it into the output space (in this case, 2 classes: spam and ham).
    *   **`num_labels=2`**: Configures the classification layer to output logits for 2 classes. Logits are raw, unnormalized scores that represent the model's confidence for each class.

**4. Dataset Preparation for BERT**

To effectively use BERT, the data needs to be formatted in a way that BERT and the `transformers` library expect. This involves creating a custom dataset class.

```python
class SpamHamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = SpamHamDataset(SpamHam['clean_text'].values, SpamHam['label_num'].values, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

*   **`class SpamHamDataset(Dataset): ...`**: Defines a PyTorch `Dataset` class to handle the spam/ham data in a BERT-compatible format.
    *   **`__init__(self, texts, labels, tokenizer, max_len)`**: Constructor to initialize the dataset with texts, labels, tokenizer, and maximum sequence length (`max_len`).
    *   **`__len__(self)`**: Returns the total number of samples in the dataset.
    *   **`__getitem__(self, idx)`**:  This is the crucial method for fetching a single data sample at a given index `idx`.
        *   **`text = self.texts[idx]`**, **`label = self.labels[idx]`**: Retrieves the text and label for the given index.
        *   **`encoding = self.tokenizer.encode_plus(...)`**:  This is where the text is processed by the BERT tokenizer.
            *   **`text`**: The input text to be tokenized.
            *   **`add_special_tokens=True`**:  Adds `[CLS]` and `[SEP]` tokens to the sequence.
            *   **`max_length=self.max_len`**:  Truncates sequences longer than `max_len` (here, 128 tokens).
            *   **`truncation=True`**: Enables truncation.
            *   **`return_token_type_ids=False`**:  For single-sentence tasks, token type IDs are not needed, so we disable them.
            *   **`padding='max_length'`**:  Pads shorter sequences to `max_len` with padding tokens (`[PAD]`).
            *   **`return_attention_mask=True`**:  Creates an attention mask to tell BERT to ignore padding tokens during attention calculations.
            *   **`return_tensors='pt'`**:  Returns PyTorch tensors.
            *   **Output `encoding`**:  This dictionary contains:
                *   **`input_ids`**: A tensor of token IDs representing the input text.  Each token is mapped to its corresponding index in the tokenizer's vocabulary.
                *   **`attention_mask`**: A tensor indicating which tokens are actual words (1) and which are padding tokens (0).
        *   **`return { ... }`**: Returns a dictionary containing:
            *   **`'text'`**: The original text (for reference).
            *   **`'input_ids'`**: The token IDs, flattened to a 1D tensor.
            *   **`'attention_mask'`**: The attention mask, flattened to a 1D tensor.
            *   **`'labels'`**: The label tensor, converted to `torch.long` dtype.

*   **`train_dataset = SpamHamDataset(...)`**: Creates an instance of `SpamHamDataset` for the training data, using the cleaned text, labels, tokenizer, and `max_len`.
*   **`train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)`**: Creates a PyTorch `DataLoader` to efficiently load data in batches during training.
    *   **`batch_size=16`**: Processes data in batches of 16 samples.
    *   **`shuffle=True`**: Shuffles the training data at the beginning of each epoch to improve training and prevent the model from learning the order of the data.

**5. Model Training (Fine-tuning BERT)**

The pre-trained BERT model is fine-tuned on the spam/ham dataset to adapt it specifically for this classification task.

```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

*   **`training_args = TrainingArguments(...)`**: Configures the training process.
    *   **`output_dir='./results'`**: Directory to save model checkpoints and predictions.
    *   **`num_train_epochs=3`**: Number of times to iterate over the entire training dataset (epochs). 3 epochs is a common starting point for fine-tuning BERT.
    *   **`per_device_train_batch_size=128`**: Batch size per GPU during training. Combined with gradient accumulation (if used), this determines the effective batch size.
    *   **`warmup_steps=500`**: Number of steps for learning rate warmup.  During warmup, the learning rate is gradually increased from zero to the initial learning rate. This helps stabilize training, especially at the beginning.
    *   **`weight_decay=0.01`**:  L2 regularization strength to prevent overfitting.
    *   **`logging_dir='./logs'`**: Directory to save training logs.

*   **`trainer = Trainer(...)`**: Creates a `Trainer` object to manage the training loop.
    *   **`model=model`**:  The BERT model (`BertForSequenceClassification`) to be trained.
    *   **`args=training_args`**: The training configuration.
    *   **`train_dataset=train_dataset`**: The training dataset (`SpamHamDataset`).
    *   **Implicit Training Loop**: The `Trainer` handles the entire training loop, including:
        *   **Forward Pass**: Feeding batches of data through the model to get logits.
        *   **Loss Calculation**: Computing the loss between the model's predictions (logits) and the true labels. For sequence classification, the `BertForSequenceClassification` model typically uses Cross-Entropy Loss.  Let $L$ be the loss function, $y_i$ be the true label for sample $i$, and $\hat{y}_i$ be the model's predicted probability distribution over classes for sample $i$. The loss for a batch can be represented as:
            $Loss = -\frac{1}{N_{batch}} \sum_{i \in batch} \sum_{j \in classes} y_{ij} \log(\hat{y}_{ij})$ where $y_{ij}=1$ if sample $i$ belongs to class $j$ and 0 otherwise, and $\hat{y}_{ij}$ is the predicted probability of sample $i$ belonging to class $j$.
        *   **Backward Pass**: Calculating gradients of the loss with respect to the model's parameters using backpropagation.
        *   **Optimization**: Updating model parameters using an optimizer (AdamW is commonly used with BERT) to minimize the loss. The optimization step can be represented as:
            $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ where $\theta_t$ are the model parameters at step $t$, $\eta$ is the learning rate, and $\nabla L(\theta_t)$ is the gradient of the loss.
        *   **Logging and Checkpointing**: Saving training progress and model checkpoints.

*   **`trainer.train()`**: Starts the fine-tuning process.

**6. Model Evaluation**

After fine-tuning, the model is evaluated on the same dataset (though in a real-world scenario, one would use a separate test set) to assess its performance.

```python
# %%
from sklearn.metrics import classification_report, confusion_matrix

test_dataset = SpamHamDataset(SpamHam['clean_text'].values, SpamHam['label_num'].values, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
predictions, true_labels = [], []

for batch in test_loader:
    inputs = batch['input_ids'].to('cuda')
    masks = batch['attention_mask'].to('cuda')
    labels = batch['labels'].to('cuda')

    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

print('Classification Report:')
print(classification_report(true_labels, predictions))
print('Confusion Matrix:')
print(confusion_matrix(true_labels, predictions))
```

*   **`test_dataset = SpamHamDataset(...)`**, **`test_loader = DataLoader(...)`**: Creates a `SpamHamDataset` and `DataLoader` for evaluation data, similar to the training setup, but `shuffle=False` for evaluation.
*   **`model.eval()`**: Sets the model to evaluation mode. This is important because it deactivates dropout layers and batch normalization layers, ensuring consistent predictions during evaluation.
*   **`predictions, true_labels = [], []`**: Initializes lists to store predictions and true labels.
*   **`for batch in test_loader:`**: Iterates through the test data in batches.
    *   **`inputs = batch['input_ids'].to('cuda')`**, **`masks = batch['attention_mask'].to('cuda')`**, **`labels = batch['labels'].to('cuda')`**: Moves input data to the GPU (`'cuda'`) if available, for faster computation.
    *   **`with torch.no_grad(): ...`**:  Disables gradient calculations during evaluation, as we only need to perform a forward pass and not update model parameters. This saves memory and computation time.
    *   **`outputs = model(inputs, attention_mask=masks)`**: Performs the forward pass through the BERT model. The output `outputs` is an object containing the model's output, including `logits`.
    *   **`logits = outputs.logits`**: Extracts the logits (raw scores) from the model output.  For a classification task with $n_c$ classes, the logits for each sample will be a vector of size $n_c$.
    *   **`predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())`**: Converts logits to predicted class labels.
        *   **`torch.argmax(logits, dim=1)`**: For each sample, finds the index of the maximum value in the logits vector along dimension 1 (across classes). This index represents the predicted class label. Mathematically, if $l_{ij}$ is the logit for sample $i$ and class $j$, the predicted class $c_i^*$ for sample $i$ is $c_i^* = \underset{j}{\operatorname{argmax}} l_{ij}$.
        *   **.cpu().numpy()**: Moves the predicted labels to the CPU and converts them to a NumPy array.
        *   **`predictions.extend(...)`**: Adds the predicted labels to the `predictions` list.
    *   **`true_labels.extend(labels.cpu().numpy())`**:  Moves true labels to CPU and adds them to the `true_labels` list.

*   **`print('Classification Report:')`**, **`print(classification_report(true_labels, predictions))`**:  Prints a classification report using scikit-learn's `classification_report` function. This report provides key metrics like precision, recall, F1-score, and support for each class, as well as overall accuracy.
*   **`print('Confusion Matrix:')`**, **`print(confusion_matrix(true_labels, predictions))`**: Prints the confusion matrix using scikit-learn's `confusion_matrix` function. The confusion matrix visualizes the performance by showing counts of true positives, true negatives, false positives, and false negatives.

**7. Heart of Transformers/BERT: Self-Attention**

The power of BERT comes from its Transformer architecture and, in particular, the **self-attention mechanism**.

*   **Limitations of Bag-of-Words and RNNs**: Traditional methods like Bag-of-Words models ignore word order and context. Recurrent Neural Networks (RNNs), while processing sequences, can struggle with long-range dependencies and parallelization.
*   **Transformers and Self-Attention**: Transformers address these limitations by using self-attention. Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word. In essence, when BERT processes a word, it looks at all *other* words in the sentence and determines how relevant they are to the current word.
*   **Contextual Embeddings**:  Self-attention enables BERT to generate **contextualized word embeddings**.  Unlike static word embeddings (like Word2Vec or GloVe) where a word has a fixed vector representation regardless of context, BERT's embeddings for a word change depending on the sentence it appears in. For example, the word "bank" in "river bank" and "bank account" will have different embeddings in BERT because the surrounding words (context) are different.
*   **Bidirectional Understanding**: BERT is **bidirectional**, meaning it considers context from both left and right sides of a word simultaneously. This is achieved through the Masked Language Modeling (MLM) pre-training objective, where BERT is trained to predict masked words in a sentence using information from both directions.
*   **Fine-tuning**:  Pre-trained BERT models have learned rich language representations. Fine-tuning adapts these general representations to a specific task (like spam/ham classification) using a smaller task-specific dataset. This is much more efficient than training a complex model from scratch.

**In mathematical terms (simplified view of self-attention for a single word):**

For a given word in the input sequence, self-attention calculates an "attention weight" for every other word in the sequence. These weights determine how much each other word contributes to the representation of the current word.  This can be conceptually represented as a weighted sum of value vectors, where the weights are determined by the attention mechanism based on queries and keys derived from the input word embeddings.

**8. Conclusion**

This report has detailed the implementation of a spam/ham classifier using BERT. By leveraging pre-trained language representations and the self-attention mechanism, BERT offers a significant advancement over traditional methods. The fine-tuning process allows adaptation to the specific nuances of the spam/ham classification task, leading to potentially higher accuracy and a better understanding of the textual content of emails compared to simpler models like Naive Bayes. The use of the `transformers` library simplifies the process of using and fine-tuning complex models like BERT, making state-of-the-art NLP techniques more accessible.