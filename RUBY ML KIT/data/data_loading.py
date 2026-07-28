# Creates tabular csv of N / 60K MNIST images

# Dwight Mayer, July 20th, 2026
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd

n_train = 250
n_test = 50
random_state = 2650

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype("float32") / 255.0
y = mnist.target.astype("int")

X_subset, _, y_subset, _ = train_test_split(
    X, y,
    train_size=n_train + n_test,
    stratify=y,
    random_state=random_state
)

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset,
    train_size=n_train,
    test_size=n_test,
    stratify=y_subset,
    random_state=random_state
)

train_df = pd.DataFrame(X_train)
train_df.insert(0, "label", y_train)
train_df.to_csv("data/mnist_train_small.csv", index=False)

test_df = pd.DataFrame(X_test)
test_df.insert(0, "label", y_test)
test_df.to_csv("data/mnist_test_small.csv", index=False)

print(f"wrote {len(train_df)} training examples to mnist_train_small.csv")
print(f"wrote {len(test_df)} test examples to mnist_test_small.csv")

