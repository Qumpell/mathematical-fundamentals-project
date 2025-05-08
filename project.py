import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesContinuous:
    def __init__(self, kernel_smoothing=False):
        self.kernel_smoothing = kernel_smoothing
        self.classes_ = None
        self.class_priors_ = None
        self.models_ = {}  # to store parameters per class and feature

    def fit(self, X, y, priors=None):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.class_priors_ = {}

        for cls in self.classes_:
            X_c = X[y == cls]
            self.models_[cls] = []
            if priors is None:
                self.class_priors_[cls] = len(X_c) / n_samples
            else:
                self.class_priors_[cls] = priors[cls]
            
            for feature in range(n_features):
                if self.kernel_smoothing:
                    kde = gaussian_kde(X_c[:, feature])
                    self.models_[cls].append(kde)
                else:
                    mean = X_c[:, feature].mean()
                    std = X_c[:, feature].std()
                    self.models_[cls].append((mean, std))

    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = {}
            for cls in self.classes_:
                prior = self.class_priors_[cls]
                likelihood = 1.0
                for i, feature_value in enumerate(x):
                    if self.kernel_smoothing:
                        kde = self.models_[cls][i]
                        likelihood *= kde.evaluate([feature_value])[0]
                    else:
                        mean, std = self.models_[cls][i]
                        likelihood *= norm.pdf(feature_value, mean, std)
                class_probs[cls] = prior * likelihood
            # normalize
            total = sum(class_probs.values())
            for cls in class_probs:
                class_probs[cls] /= total
            probs.append(class_probs)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([max(p, key=p.get) for p in probs])

    def plot_decision_boundary(self, X, y, resolution=0.02):
        if X.shape[1] != 2:
            raise ValueError("Visualization works only for 2D features.")

        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=30)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Naive Bayes Decision Boundary")
        plt.legend(handles=scatter.legend_elements()[0], labels=[str(cls) for cls in self.classes_])
        plt.show()

# Przyklad uzycia
if __name__ == "__main__":
    # Wczytywanie danych Pima Indians Diabetes Dataset
    import pandas as pd
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=col_names)

    X = df.iloc[:, :-1].values  # Wszystkie kolumny poza 'Outcome'
    y = df['Outcome'].values    # Kolumna 'Outcome'

    # Skalowanie danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Trenowanie modelu
    nb = NaiveBayesContinuous(kernel_smoothing=False)
    nb.fit(X_train, y_train)

    # Predykcja i ocena modelu
    y_pred = nb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Wizualizacja (na 2 cechach: Glucose vs BMI)
    X_2d = X_scaled[:, [1, 5]]  # Glucose (1) i BMI (5)
    y_2d = y

    nb_2d = NaiveBayesContinuous(kernel_smoothing=False)
    nb_2d.fit(X_2d, y_2d)
    nb_2d.plot_decision_boundary(X_2d, y_2d)
