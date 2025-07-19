import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay

def plot_crossval_roc(clf, X, y, label, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        aucs.append(auc(fpr, tpr))
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(base_fpr, mean_tpr, label=f"{label} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")


# Naloži podatke
df = pd.read_csv("rezultati_suv.csv")

# Razdeli v dve skupini
df_ad = df[df["label"] == 1]
df_cn = df[df["label"] == 0]

# Primer: analiza za SUVmean v precuneusu
feature = "SUVmean_precuneus"
t_stat, t_p = ttest_ind(df_ad[feature], df_cn[feature])
u_stat, u_p = mannwhitneyu(df_ad[feature], df_cn[feature])

print(f"T-test za {feature}: p = {t_p:.4f}")
print(f"Mann-Whitney U-test za {feature}: p = {u_p:.4f}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# Značilnosti in ciljna spremenljivka
features = [
    "SUVmean_brain", "SUVpeak_brain", "SUVmax_brain",
    "SUVmean_precuneus", "SUVpeak_precuneus", "SUVmax_precuneus",
    "SUVmean_cerebellum", "SUVpeak_cerebellum", "SUVmax_cerebellum"
]
X = df[features]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Normalizacija (popravljeno)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# Logistična regresija
clf_lr = LogisticRegression()
clf_lr.fit(X_train_scaled, y_train)
y_pred_lr = clf_lr.predict(X_test_scaled)
print("Logistična regresija - Accuracy:", accuracy_score(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, clf_lr.predict_proba(X_test_scaled)[:, 1]))

# Odločilno drevo
clf_tree = DecisionTreeClassifier(max_depth=3)
clf_tree.fit(X_train, y_train)
print("Odločilno drevo - Accuracy:", accuracy_score(y_test, clf_tree.predict(X_test)))

# SVM
clf_svm = SVC(probability=True)
clf_svm.fit(X_train_scaled, y_train)
print("SVM - Accuracy:", accuracy_score(y_test, clf_svm.predict(X_test)))
print("SVM - AUC:", roc_auc_score(y_test, clf_svm.predict_proba(X_test)[:, 1]))

print("\n--- Statistična analiza ---")
for feature in features:
    t_stat, t_p = ttest_ind(df_ad[feature], df_cn[feature])
    u_stat, u_p = mannwhitneyu(df_ad[feature], df_cn[feature])
    print(f"{feature}")
    print(f"  T-test p = {t_p:.4f} | U-test p = {u_p:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot za vse značilnosti
print("\n--- Boxploti ---")
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="label", y=feature)
    plt.title(f"Boxplot: {feature}")
    plt.xlabel("Skupina (0 = CN, 1 = AD)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"boxplot_{feature}.png")  # shraniš tudi kot sliko
    plt.close()


from sklearn.metrics import roc_curve, auc

# Napovedi za ROC krivulje
y_score_lr = clf_lr.predict_proba(X_test_scaled)[:, 1]
y_score_svm = clf_svm.predict_proba(X_test_scaled)[:, 1]
y_score_tree = clf_tree.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_score_tree)

# Izriši ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistična regresija (AUC = {auc(fpr_lr, tpr_lr):.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc(fpr_svm, tpr_svm):.2f})')
plt.plot(fpr_tree, tpr_tree, label=f'Drevo (AUC = {auc(fpr_tree, tpr_tree):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC krivulje za klasifikatorje')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_krivulje.png")
plt.show()

from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

print("\n--- Cross-validacija (ROC AUC) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validated AUC za vse tri modele
auc_lr = cross_val_score(clf_lr, X[features], y, cv=cv, scoring='roc_auc')
auc_svm = cross_val_score(clf_svm, X[features], y, cv=cv, scoring='roc_auc')
auc_tree = cross_val_score(clf_tree, X[features], y, cv=cv, scoring='roc_auc')

print(f"Logistična regresija - Povprečen AUC: {np.mean(auc_lr):.4f} ± {np.std(auc_lr):.4f}")
print(f"SVM - Povprečen AUC: {np.mean(auc_svm):.4f} ± {np.std(auc_svm):.4f}")
print(f"Odločilno drevo - Povprečen AUC: {np.mean(auc_tree):.4f} ± {np.std(auc_tree):.4f}")


# Povprečne ROC krivulje za vse klasifikatorje
plt.figure(figsize=(7, 6))
plot_crossval_roc(LogisticRegression(), X[features], y, "Logistična regresija")
plot_crossval_roc(SVC(probability=True), X[features], y, "SVM")
plot_crossval_roc(DecisionTreeClassifier(max_depth=3), X[features], y, "Odločilno drevo")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Povprečne ROC krivulje (5-fold CV)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_krivulje_cv.png")
plt.show()

print("\n--- Violin + Boxploti za vse značilnosti ---")
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="label", y=feature, inner=None, color=".8")
    sns.boxplot(data=df, x="label", y=feature, width=0.3)
    plt.title(f"{feature}")
    plt.xlabel("Skupina (0 = CN, 1 = AD)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"violin_boxplot_{feature}.png")  # Shrani sliko
    plt.show()  # Prikaže sliko

