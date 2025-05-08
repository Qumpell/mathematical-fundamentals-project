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

    def plot_decision_boundary_2d(self, X, y, resolution=0.2, test_point=None):
        if X.shape[1] != 2:
            raise ValueError("Wizualizacja w tej metodzie działa tylko dla 2 cech!")

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
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)

        if test_point is not None:
            predicted_class = self.predict([test_point])[0]
            print(f"Punkt testowy został zaklasyfikowany do klasy: {predicted_class}")
            plt.scatter(test_point[0], test_point[1], c='black', s=100, marker='X', label="Punkt testowy")


        plt.xlabel("Poziom glukozy (standaryzowany)", fontsize=12)
        plt.ylabel("BMI (standaryzowany)", fontsize=12)
        plt.title("Granica decyzyjna - cechy: Glukoza, BMI", fontsize=14)

        handles, _ = scatter.legend_elements()
        labels = [f"Klasa {cls} ({'chory' if cls == 1 else 'zdrowy'})" for cls in self.classes_]
        plt.legend(handles=handles, labels=labels, title="Legenda", fontsize=10, title_fontsize=11)
        
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()


    def plot_decision_boundary_3d(self, X, y, resolution=0.2, test_point=None):
        if X.shape[1] != 3:
            raise ValueError("Wizualizacja w tej metodzie działa tylko dla 3 cech!")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(
            np.arange(x_min, x_max, resolution),
            np.arange(y_min, y_max, resolution),
            np.arange(z_min, z_max, resolution)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = self.predict(grid_points)
        Z = np.array(Z)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Kolory klas
        colors = np.array(['#FF0000', '#0000FF', '#00FF00'])
        color_vals = colors[Z.astype(int)] if len(self.classes_) <= 3 else Z

        ax.scatter(
            grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
            c=color_vals, alpha=0.08, s=2, label="Obszary decyzyjne"
        )

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=60, label="Dane treningowe")

        if test_point is not None:
            predicted_class = self.predict([test_point])[0]
            print(f"Punkt testowy został zaklasyfikowany do klasy: {predicted_class}")
            ax.scatter(test_point[0], test_point[1], test_point[2],
                       c='black', s=100, marker='X', label="Punkt testowy")

        ax.set_xlabel("Poziom glukozy (standaryzowany)")
        ax.set_ylabel("BMI (standaryzowany)")
        ax.set_zlabel("Wiek (standaryzowany)")
        ax.set_title("Granica decyzyjna - cechy: Glukoza, BMI, Wiek")
        ax.legend()
        plt.tight_layout()
        plt.show()

# Przyklad uzycia
if __name__ == "__main__":
    import pandas as pd
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, names=col_names)

    X = df.iloc[:, :-1].values
    y = df['Outcome'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    nb = NaiveBayesContinuous(kernel_smoothing=False)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Wykres 2D (Glucose i BMI)
    print("Generowanie wykresu 2D...")
    X_2d = X_scaled[:, [1, 5]]
    nb_2d = NaiveBayesContinuous(kernel_smoothing=False)
    nb_2d.fit(X_2d, y)
    test_point_2d = X_2d[0]
    nb_2d.plot_decision_boundary_2d(X_2d, y, test_point=test_point_2d)

    # Wykres 3D (Glucose, BMI, Age)
    print("Generowanie wykresu 3D...")
    X_3d = X_scaled[:, [1, 5, 7]]
    nb_3d = NaiveBayesContinuous(kernel_smoothing=False)
    nb_3d.fit(X_3d, y)
    test_point_3d = X_3d[0]
    nb_3d.plot_decision_boundary_3d(X_3d, y, test_point=test_point_3d)
