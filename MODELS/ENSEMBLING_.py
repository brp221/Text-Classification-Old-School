#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:39:39 2023

@author: bratislavpetkovic
"""

# ENSEMBLING 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statistics
import numpy as np 

predictions_cnn_lstm = model_RCNN.predict(X_test)
predictions_cnn_lstm_labels = np.argmax(predictions_cnn_lstm, axis=1)
test_report_lstm = classification_report(y_test, predictions_cnn_lstm_labels)
print(test_report_lstm)

predictions_cnn_2d = cnn_2d_model.predict(X_test)
predictions_cnn_2d_labels = np.argmax(predictions_cnn_2d, axis=1)
test_report_cnn = classification_report(y_test, predictions_cnn_2d_labels)
print(test_report_cnn)

predictions_svm = svm_model.predict_proba(X_test)
predictions_svm_labels = np.argmax(predictions_svm, axis=1)
test_report_svm = classification_report(y_test, predictions_svm_labels   )
print(test_report_svm)


# Majority Voting Ensemble
test_3_major_vote_labels = [statistics.mode(votes) for votes in list(zip(predictions_cnn_2d_labels, predictions_cnn_lstm_labels, predictions_svm_labels))]
test_report_3_maj_vote_report = classification_report(y_test, test_3_major_vote_labels   )
print(test_report_3_maj_vote_report)


# Averaging Votes Ensemble
test_3_preds_average = (predictions_cnn_2d + predictions_svm + predictions_cnn_lstm) /3
test_3_preds_average_labels = np.argmax(test_3_preds_average, axis=1)
test_3_preds_average_report = classification_report(y_test, test_3_preds_average_labels   )
print(test_3_preds_average_report)


