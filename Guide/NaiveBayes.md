```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter
        self.class_log_priors = None # Log of class priors P(c)
        self.feature_log_probs = None # Log of feature conditional probabilities P(w|c)
        self.classes = None # Unique class labels

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = torch.unique(y)
        n_classes = len(self.classes)

        class_counts = torch.zeros(n_classes)
        for i, c in enumerate(self.classes):
            class_counts[i] = (y == c).sum()
        self.class_log_priors = torch.log(class_counts) - torch.log(torch.tensor(n_samples))

        feature_counts = torch.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            feature_counts[i] = X[y == c].sum(dim=0) + self.alpha

        feature_totals = torch.sum(feature_counts, dim=1, keepdim=True)
        self.feature_log_probs = torch.log(feature_counts) - torch.log(feature_totals)

        return self
```

**Initialization (`__init__`)**:

* `self.alpha = alpha`:
   Let $\alpha$ be the smoothing parameter (Laplace smoothing).

* `self.class_log_priors = None`:
   Initialize `class_log_priors` to store the logarithm of the prior probability for each class, denoted as $\log P(c)$, where $c$ represents a class.

* `self.feature_log_probs = None`:
   Initialize `feature_log_probs` to store the logarithm of the conditional probability of each feature (word) given a class, denoted as $\log P(w|c)$, where $w$ is a word and $c$ is a class.

* `self.classes = None`:
   Initialize `classes` to store the set of unique class labels.


**Fitting (`fit` function)**:

* `n_samples, n_features = X.shape`:
   Let $N$ be the total number of samples (emails) and $F$ be the number of features (vocabulary size).

* `self.classes = torch.unique(y)`:
   Let $C$ be the set of unique classes, obtained from the labels $y$.

* `n_classes = len(self.classes)`:
   Let $n_c$ be the number of classes, i.e., $n_c = |C|$.

* ```python
  class_counts = torch.zeros(n_classes)
  for i, c in enumerate(self.classes):
      class_counts[i] = (y == c).sum()
  ```
  For each class $c_i \in C$, calculate the count of samples belonging to class $c_i$, denoted as $Count(c_i)$. Let $ClassCounts$ be a vector where $ClassCounts_i = Count(c_i)$.

* `self.class_log_priors = torch.log(class_counts) - torch.log(torch.tensor(n_samples))`
   Calculate the log prior probability for each class $c_i$:
   $$ \log P(c_i) = \log \left( \frac{Count(c_i)}{N} \right) = \log(Count(c_i)) - \log(N) $$
   Store these log prior probabilities in `self.class_log_priors`.

* ```python
  feature_counts = torch.zeros((n_classes, n_features))
  for i, c in enumerate(self.classes):
      feature_counts[i] = X[y == c].sum(dim=0) + self.alpha
  ```
  For each class $c_i \in C$ and each feature $j$ (word $w_j$ in vocabulary), calculate the count of feature $j$ in all samples belonging to class $c_i$, and apply Laplace smoothing. Let $Count(w_j, c_i)$ be the count of word $w_j$ in documents of class $c_i$. Then, the smoothed feature count for word $w_j$ and class $c_i$ is $SmoothedCount(w_j, c_i) = Count(w_j, c_i) + \alpha$. Let $FeatureCounts$ be a matrix where $FeatureCounts_{ij} = SmoothedCount(w_j, c_i)$.

* ```python
  feature_totals = torch.sum(feature_counts, dim=1, keepdim=True)
  self.feature_log_probs = torch.log(feature_counts) - torch.log(feature_totals)
  ```
  For each class $c_i$, calculate the total count of all features (words) in class $c_i$ after smoothing. Let $TotalWordCount(c_i) = \sum_{j=1}^{F} SmoothedCount(w_j, c_i)$.
  Then, calculate the log conditional probability for each word $w_j$ given class $c_i$:
   $$ \log P(w_j|c_i) = \log \left( \frac{SmoothedCount(w_j, c_i)}{TotalWordCount(c_i)} \right) = \log(SmoothedCount(w_j, c_i)) - \log(TotalWordCount(c_i)) $$
   Store these log conditional probabilities in `self.feature_log_probs`.

* `return self`: Returns the fitted model.


```python
    def predict_log_proba(self, X):
        joint_log_likelihood = torch.zeros((X.shape[0], len(self.classes)))

        for i, c in enumerate(self.classes):
            joint_log_likelihood[:, i] = self.class_log_priors[i] + torch.matmul(X, self.feature_log_probs[i])

        return joint_log_likelihood
```

**Predict Log Probability (`predict_log_proba` function)**:

* `joint_log_likelihood = torch.zeros((X.shape[0], len(self.classes)))`:
   Initialize a matrix `joint_log_likelihood` to store the joint log-likelihood for each sample and each class. Let $M$ be the number of test samples. The matrix will have dimensions $M \times n_c$.

* ```python
  for i, c in enumerate(self.classes):
      joint_log_likelihood[:, i] = self.class_log_priors[i] + torch.matmul(X, self.feature_log_probs[i])
  ```
  For each class $c_i \in C$:
  For each test sample $x_k$ (represented as a Bag-of-Words vector in row $k$ of $X$), calculate the joint log-likelihood of sample $x_k$ belonging to class $c_i$.
  According to Naive Bayes assumption, the probability of observing features $x_k = (x_{k1}, x_{k2}, ..., x_{kF})$ given class $c_i$ is $P(x_k|c_i) = \prod_{j=1}^{F} P(w_j|c_i)^{x_{kj}}$, where $x_{kj}$ is the count of word $w_j$ in sample $x_k$.

  The joint probability is $P(x_k, c_i) = P(c_i) \cdot P(x_k|c_i) = P(c_i) \cdot \prod_{j=1}^{F} P(w_j|c_i)^{x_{kj}}$.

  Taking the logarithm of the joint probability:
  $$ \log P(x_k, c_i) = \log \left( P(c_i) \cdot \prod_{j=1}^{F} P(w_j|c_i)^{x_{kj}} \right) = \log P(c_i) + \sum_{j=1}^{F} \log \left( P(w_j|c_i)^{x_{kj}} \right) = \log P(c_i) + \sum_{j=1}^{F} x_{kj} \cdot \log P(w_j|c_i) $$

  In the code:
  * `self.class_log_priors[i]` is $\log P(c_i)$.
  * `self.feature_log_probs[i]` is a vector of $\log P(w_j|c_i)$ for all words $w_j$.
  * `torch.matmul(X, self.feature_log_probs[i])` calculates $\sum_{j=1}^{F} x_{kj} \cdot \log P(w_j|c_i)$ for each sample $k$.
  * `joint_log_likelihood[:, i]` stores the calculated $\log P(x_k, c_i)$ for all samples $k$ for class $c_i$.

* `return joint_log_likelihood`: Returns the matrix of joint log-likelihoods.


```python
    def predict(self, X):
        return self.classes[torch.argmax(self.predict_log_proba(X), dim=1)]
```

**Prediction (`predict` function)**:

* `return self.classes[torch.argmax(self.predict_log_proba(X), dim=1)]`:
   For each test sample $x_k$, predict the class that maximizes the joint log-likelihood (which is equivalent to maximizing the posterior probability since the evidence $P(x_k)$ is the same for all classes when comparing classes for a single sample).

   For each sample $x_k$, find the class $c^*$ that maximizes $\log P(x_k, c_i)$:
   $$ c^* = \underset{c_i \in C}{\operatorname{argmax}} \log P(x_k, c_i) $$

   In the code:
   * `self.predict_log_proba(X)` returns the matrix of joint log-likelihoods.
   * `torch.argmax(..., dim=1)` finds the index of the class with the maximum log-likelihood for each sample along dimension 1 (across classes).
   * `self.classes[...]` uses these indices to retrieve the corresponding class labels from `self.classes`.

In summary, the code implements Multinomial Naive Bayes by:
1. **Estimating class priors** $P(c_i)$ and **feature conditional probabilities** $P(w_j|c_i)$ from the training data using smoothed counts and storing their logarithms.
2. **For prediction**: For each test sample, calculating the joint log-likelihood $\log P(x_k, c_i)$ for each class $c_i$.
3. **Assigning the class** with the highest joint log-likelihood as the predicted class.