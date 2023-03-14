"""
Support Vector Machine
"""

from sklearn.svm import SVC
from sklearn.metrics import classification_report


def support_vector_machine(x_train, y_train, x_test, y_test):
    model = SVC(degree=5)
    model.fit(x_train, y_train["label"])

    y_pred = model.predict(x_test)
    report = classification_report(y_test["label"], y_pred, output_dict=True)

    return model, report
