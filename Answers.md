# Performance Metrics for Classification - Answer Sheet

---

## Question 1: Classification and Classifiers

> **Summarize in your own words, based on the Wikipedia content quoted above, and potentially additional sources:**

### What are we trying to solve in a classification problem?

In a classification problem, we are trying to determine which category (or class) a given observation belongs to, based on its measurable properties (known as features). The core task is to take an input — described by a set of quantifiable characteristics — and assign it to one of a predefined set of categories.

For example, given an email's text content and metadata (the features), we want to decide whether it belongs to the "spam" or "non-spam" category. Similarly, given a patient's medical observations such as blood pressure, sex, and symptoms, we want to assign a diagnosis. In the MNIST dataset used in the notebook, the task is to look at pixel intensity values of a handwritten digit image and determine which digit (0 through 9) it represents — or, in the simplified binary version, whether the digit is a "5" or "not a 5."

Features used for classification can take many forms: categorical (e.g., blood type: A, B, AB, O), ordinal (e.g., size: small, medium, large), integer-valued (e.g., word frequency counts), or real-valued (e.g., blood pressure measurements). The classification algorithm must learn the relationship between these features and the target categories from labeled training data, and then generalize that knowledge to correctly categorize new, unseen observations.

### What is a classifier?

A classifier is an algorithm (or its concrete implementation) that performs classification — that is, it takes input data and maps it to a predicted category. The term can refer to two related concepts:

1. **The algorithm or system**: The concrete implementation that accepts input features and produces a class prediction. For instance, the `SGDClassifier` used in the notebook is a classifier that trains a linear model using stochastic gradient descent and then predicts whether a digit image is a "5" or not.

2. **The mathematical function**: The underlying learned function that maps input data to a category. After training, a classifier has internalized a decision boundary (or decision function) that separates the feature space into regions corresponding to different classes. New observations are classified based on which region they fall into.

In practice, a classifier is trained on labeled examples (where the correct category is known), learns patterns that distinguish the categories, and then applies those patterns to assign categories to new, unlabeled observations.

---

## Question 2: Preprocessing — Boolean Values in `y_train_5`

> **What does the Boolean values (`True` and `False`) stand for in the target `y_train_5`?**

In `y_train_5`, each Boolean value indicates whether the corresponding digit image is a `'5'` or not:

- `True` means the original label in `y_train` was `'5'` — this image belongs to the **positive class**.
- `False` means the original label was any other digit (`'0'` through `'4'`, or `'6'` through `'9'`) — this image belongs to the **negative class**.

This transformation converts the original 10-class multiclass problem into a simpler binary classification problem: "Is this digit a 5, or not?"

> **Why wasn't `X` changed as well?**

`X` contains the feature data (the 784 pixel intensity values for each image). The features themselves do not change — the images are still the same images. What changed is only how we label them. We are simply re-framing the question we ask the model: instead of "which of 10 digits is this?", we now ask "is this a 5 or not?". The input data (pixel values) remains exactly the same; only the target labels needed to be transformed to reflect the new binary question.

---

## Question 3: The `predict` Result for `some_digit`

> **Which image does `some_digit` stand for?**

`some_digit` is set to `X[2]`, the third image in the dataset. When plotted, it shows the handwritten digit **4**. Its true label is `y[2]`.

> **What is the meaning of the resulting Boolean value (`True` and `False`)?**

The classifier returns `False` for this image, meaning it predicts that this digit is **not a 5**. Since the image is indeed a `4` (not a `5`), this is a correct prediction — specifically, a **True Negative**. If the classifier had returned `True`, it would have predicted the digit is a `5`.

---

## Question 4: The `cross_validate` Function

> **Describe in your own words what the `cv` argument is for.**

The `cv` argument specifies the cross-validation splitting strategy. When `cv=3`, the training data is divided into 3 equal-sized, non-overlapping subsets called "folds." The model is then trained and evaluated 3 times: each time, one fold is held out as the validation (test) set, and the remaining two folds are used for training. This rotation ensures every data point is used for validation exactly once, providing a more reliable and less biased estimate of model performance than a single train/test split.

> **How many times has the model been trained?**

The model has been trained **3 times** — once per fold. In each iteration, the model is trained from scratch on 2 of the 3 folds and evaluated on the remaining fold.

> **How is the value of `cv` related to the output of this function?**

The value of `cv` determines the number of elements in each output array. Since `cv=3`, each array in the output dictionary (`fit_time`, `score_time`, `test_score`) contains exactly 3 values — one result from each of the 3 cross-validation iterations.

> **What do `fit_time`, `score_time` and `test_score` stand for?**

- **`fit_time`**: An array of the time (in seconds) it took to train the model for each fold. For example, `[5.56, 4.98, 3.97]` means the first fold took ~5.56 seconds to train.
- **`score_time`**: An array of the time (in seconds) it took to evaluate (score) the model on the validation fold. This measures only prediction and metric computation time, not training time.
- **`test_score`**: An array of the accuracy scores achieved on the held-out validation fold for each iteration. The values `[0.95035, 0.96035, 0.9604]` mean the model correctly classified approximately 95–96% of the samples in each validation fold.

---

## Question 5: Interpreting Accuracy

> **Could you provide a textual description of the accuracy result for the first fold?**

The accuracy for the first fold is 0.95035. Since each fold contains 20,000 samples (60,000 / 3), this means: out of 20,000 digit images, approximately **19,007 were correctly predicted** (either correctly identified as `5` or correctly identified as not `5`), while approximately **993 were incorrectly predicted** (either a `5` missed by the model, or a non-`5` wrongly labeled as `5`).

> **Given the uniform distribution of samples across all 10 digits, and given that the classifier classifies one of them versus all the others (`5` versus all others), what would be a simple handwritten rule to achieve 90% accuracy?**

A simple rule would be: **"Always predict 'not 5' for every image."** Since the digit `5` makes up only about 10% of the dataset (1 out of 10 equally distributed digits), the remaining 90% are non-`5` digits. By always predicting `False` (not a `5`), this naive rule would be correct 90% of the time, achieving ~90% accuracy — without ever looking at the image features.

> **How would you describe the goodness of classification of the `SGDClassifier` versus the expected accuracy of such a handwritten rule?**

The `SGDClassifier` achieves approximately 95–96% accuracy, which is only about 5–6 percentage points above the naive "always predict not-5" rule at ~90%. While the SGD model clearly learned something meaningful from the features, the high baseline accuracy of the naive rule reveals that accuracy alone is a misleading metric for imbalanced binary classification problems. The seemingly high accuracy of 95% is less impressive when we realize that a trivial rule that ignores all features already gets 90%.

---

## Question 6: DummyClassifier Accuracy

> **What is the accuracy of the `DummyClassifier`?**

The `DummyClassifier` achieves an accuracy of **0.90965 (~91%)** across all three folds. This is because it always predicts `False` (not a `5`), and since approximately 90.965% of the training samples are indeed not `5`, it gets that proportion correct by default.

> **How does it compare to the handwritten rule you suggested above, and to the `SGDClassifier`?**

The `DummyClassifier`'s accuracy (~91%) is essentially identical to the simple handwritten rule of "always predict not-5," confirming that both strategies are equivalent — they ignore the input features entirely and rely solely on the class imbalance. The `SGDClassifier` at ~95–96% accuracy outperforms both, demonstrating that it has genuinely learned to identify the digit `5` from the pixel features. However, the small gap (only ~5 percentage points) between a model that actually analyzes images and one that blindly guesses highlights why accuracy alone is insufficient for evaluating classifiers on imbalanced datasets. Additional metrics like precision, recall, and F1-score are needed.

---

## Question 7: Exercise — Fashion-MNIST Data Set

> **Follow the same process used for MNIST, but using Fashion-MNIST. Are you able to replicate the process? Provide your insights.**

The Fashion-MNIST dataset has the same structure as MNIST: 70,000 grayscale 28x28 images split into 60,000 training and 10,000 test samples, with 10 classes. The classes represent clothing items instead of digits: T-shirt/top (0), Trouser (1), Pullover (2), Dress (3), Coat (4), Sandal (5), Shirt (6), Sneaker (7), Bag (8), Ankle boot (9).

**Preprocessing** follows the same pattern:

```python
X_train_f, X_test_f = X_fashion[:60000], X_fashion[60000:]
y_train_f, y_test_f = y_fashion[:60000], y_fashion[60000:]

# Binary classification: "Is it a Sandal (label '5') or not?"
y_train_sandal = (y_train_f == '5')
y_test_sandal = (y_test_f == '5')
```

**Training** uses the same SGDClassifier:

```python
sgd_fashion = SGDClassifier(random_state=42)
sgd_fashion.fit(X_train_f, y_train_sandal)
```

**Evaluation** via cross-validation:

```python
cross_validate(estimator=sgd_fashion, X=X_train_f, y=y_train_sandal, cv=3, scoring="accuracy")
```

**Insights**: The process is fully replicable because Fashion-MNIST was intentionally designed as a drop-in replacement for MNIST with the same data format. However, Fashion-MNIST is generally considered a harder classification task — clothing items share more visual similarity with each other than handwritten digits do (e.g., pullovers vs. coats, shirts vs. T-shirts). We would expect the SGDClassifier to achieve similar accuracy on the binary "sandal vs. not-sandal" task (since class balance is the same at ~10%), but precision and recall may differ depending on how visually distinct sandals are from other clothing items. A DummyClassifier would again achieve ~90% accuracy, reinforcing the lesson that accuracy is misleading for imbalanced binary problems.

---

## Question 8: Confusion Matrix

> **What are the required arguments that the `confusion_matrix` function expects?**

The `confusion_matrix` function requires two arguments:
1. **`y_true`**: The ground-truth (actual) labels — what the correct classification is for each sample.
2. **`y_pred`**: The predicted labels — what the classifier predicted for each sample.

Both must be array-like and of the same length. There are also optional parameters such as `labels` (to specify the order of classes) and `normalize`.

> **What is the output of this function, specifically for a binary classifier such as the one tested here?**

For a binary classifier, the output is a **2x2 NumPy array**. In sklearn's default ordering (sorted labels: `False` first, `True` second), the matrix is laid out as:

|                      | Predicted Negative | Predicted Positive |
|----------------------|-------------------:|-------------------:|
| **Actual Negative**  | True Negatives (TN)| False Positives (FP)|
| **Actual Positive**  | False Negatives (FN)| True Positives (TP)|

> **What is the meaning of each value in both of the confusion matrices returned above?**

**SGD Classifier confusion matrix:**

```
[[53892,   687],
 [ 1891,  3530]]
```

- **53,892 (TN)**: Non-`5` digits correctly predicted as not `5`.
- **687 (FP)**: Non-`5` digits incorrectly predicted as `5` (false alarms).
- **1,891 (FN)**: Actual `5`s incorrectly predicted as not `5` (missed detections).
- **3,530 (TP)**: Actual `5`s correctly predicted as `5`.
- Total: 53,892 + 687 + 1,891 + 3,530 = 60,000 samples.

**Perfect predictions confusion matrix:**

```
[[54579,     0],
 [    0,  5421]]
```

- **54,579 (TN)**: All non-`5` digits correctly identified — no false positives.
- **0 (FP)**: No non-`5` digit was wrongly called a `5`.
- **0 (FN)**: No actual `5` was missed.
- **5,421 (TP)**: All `5`s correctly identified.
- This represents a perfect classifier with zero errors.

---

## Question 9: Exercise — Precision and Recall for the Autonomous Security Drone

> **Scenario summary:** 500 camera views, 10 actual persons present, drone raised 12 alarms, 8 correct alarms (TP), 4 false alarms (FP), 2 missed persons (FN), 488 correct non-alarms (TN).

### Calculating the Metrics

**Precision** = TP / (TP + FP) = 8 / (8 + 4) = **8/12 = 0.667 (66.7%)**

**Recall** = TP / (TP + FN) = 8 / (8 + 2) = **8/10 = 0.80 (80%)**

### Explaining the Results

- **Precision of 66.7%** means that when the drone sounds an alarm claiming a person is present, it is correct only about two-thirds of the time. One out of every three alarms is a false alarm (triggered by shadows, falling boxes, etc.). Security personnel responding to these alarms would waste effort on false triggers about 33% of the time.

- **Recall of 80%** means the drone successfully detects 8 out of the 10 actual intrusions. However, it misses 2 out of 10 real persons (20% go undetected). For a security system, this is concerning — a missed intruder could lead to a security breach. Depending on the risk tolerance, the company may want to tune the system to achieve higher recall (even if it means more false alarms), since missing a real intruder is likely more costly than responding to a false alarm.

---

## Question 10: Manual Recall Calculation (Python Code)

> **Similar to the manual calculation of precision, write down the Python code for calculating recall.**

```python
# Recall: TP / (TP + FN)
# From the confusion matrix cm:
#   cm[1, 1] = TP (True Positives)  = 3530
#   cm[1, 0] = FN (False Negatives) = 1891
cm[1, 1] / (cm[1, 0] + cm[1, 1])
# Result: 0.6511713705958311
```

This matches the output of `recall_score(y_train_5, y_train_pred)`.

---

## Question 11: Manual F1 Score Calculation (Python Code)

> **Using the definition, calculate the F1 score using True Positives (TP), False Negatives (FN) and False Positives (FP).**

```python
# From the confusion matrix:
TP = cm[1, 1]  # 3530
FN = cm[1, 0]  # 1891
FP = cm[0, 1]  # 687

# F1 = TP / (TP + (FP + FN) / 2)
f1 = TP / (TP + (FP + FN) / 2)
# Result: 0.7325171197343847
```

This matches the output of `f1_score(y_train_5, y_train_pred)`.

---

## Question 12: Precision vs. Recall — Real-World Tradeoffs

> **Medical diagnostics (first indicator for a condition): high precision or high recall?**

**High recall** is better here. In medical diagnostics used as a first screening indicator, missing a patient who actually has the condition (a False Negative) is far more dangerous than flagging a healthy patient for further review (a False Positive). A false negative means the patient may not receive necessary treatment, potentially leading to worsening health or death. A false positive simply means an additional follow-up test with a medical professional, which is a much lower cost. Therefore, we want the system to catch as many true positive cases as possible, even if it means some false alarms.

> **Stock exchange — highlighting best investment opportunities: high precision or high recall?**

**High precision** is better here. When a system recommends a stock as a good investment, investors want to be confident that the recommendation is correct. A false positive (recommending a bad stock) directly leads to financial loss. A false negative (missing a good stock) is a missed opportunity but does not cause direct harm. Investors would rather receive fewer but more reliable recommendations than be flooded with unreliable ones.

> **Two additional real-world cases:**

**High recall preferred (at the expense of precision):**
- **Airport security screening**: When scanning luggage for prohibited or dangerous items, it is critical to catch every actual threat (high recall). A false alarm means an extra manual bag inspection (inconvenient but safe). A missed threat (false negative) could have catastrophic consequences. The system should err heavily on the side of caution.

**High precision preferred (at the expense of recall):**
- **Email spam filtering**: When a spam filter marks an email as spam, users want it to be right. A false positive (a legitimate email sent to the spam folder) can cause a user to miss an important message — a job offer, a client email, or a critical notification. A false negative (a spam email reaching the inbox) is merely an annoyance. Users would rather see a few spam emails in their inbox than risk losing important messages.

---

## Question 13: Precision/Recall Behavior with Changing Threshold

> **How do precision and recall change with increasing or decreasing the threshold?**

- **Increasing the threshold**: Precision **increases** while recall **decreases**. With a higher threshold, the classifier becomes more selective — it only predicts "positive" when it is very confident. This means fewer false positives (boosting precision), but more actual positives are missed because the bar for a positive prediction is too high (lowering recall).

- **Decreasing the threshold**: Recall **increases** while precision **decreases**. With a lower threshold, the classifier predicts "positive" more liberally. This catches more actual positives (boosting recall), but also lets in more false positives since the classifier is less discriminating (lowering precision).

> **What is the intuition behind this behavior?**

The decision threshold controls how "sure" the classifier needs to be before it labels a sample as positive. Think of it as a confidence dial:

- **High threshold (strict)**: The classifier says "I'll only call something a 5 if I'm very sure." This means when it does say "5," it's usually right (high precision), but it stays silent on many actual 5s that it's not confident enough about (low recall).

- **Low threshold (lenient)**: The classifier says "I'll call anything a 5 if there's even a small chance." This catches nearly all actual 5s (high recall), but it also mistakenly labels many non-5s as 5s (low precision).

This is the fundamental **precision/recall tradeoff**: for a given trained model, you cannot improve one without sacrificing the other. The precision-recall curve visualizes all possible tradeoff points, and the threshold selection depends on the application's priorities (as discussed in Question 12).
