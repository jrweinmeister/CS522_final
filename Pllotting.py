# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:32:55 2022

@author: 14w
"""

import numpy as np
import matplotlib.pyplot as plt

# Get data for multiple features
labels = ['6', '12', '18', '24', 'Full']
rf = np.array([0.45, 0.45, 0.45, 0.433, 0.433])
svm = np.array([0.55, 0.55, 0.617, 0.617, 0.683])
knn = np.array([0.717, 0.683, 0.717, 0.717, 0.617])

fig1, ax1 = plt.subplots()
x = np.arange(len(labels))
width = 0.2
ax1.bar(x - width/2, rf, width, label='Random Forest')
ax1.bar(x + width/2, svm, width, label='Support Vector Machine')
ax1.bar(x + width*1.5, knn, width, label='K Nearest Neighbor')
ax1.set_xticks(x, labels)
ax1.set_xlabel('Sub-sample Length (s)')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
fig1.savefig('Subsamples.png', bbox_inches='tight')

# Feature type
labels = ['MFCC', 'SC', 'Band', 'MFCC, SC', 'MFCC, Band', 'SC, Band', 'MFCC, SC, Band']
rf = [0.45, 0.45, 0.417, 0.55, 0.483, 0.35, 0.45]
svm = [0.617, 0.35, 0.433, 0.65, 0.617, 0.467, 0.617]
knn = [0.517, 0.617, 0.583, 0.583, 0.65, 0.583, 0.717]

fig2, ax2 = plt.subplots()
x = np.arange(len(labels))
width = 0.2
ax2.bar(x - width/2, rf, width, label='Random Forest')
ax2.bar(x + width/2, svm, width, label='Support Vector Machine')
ax2.bar(x + width*1.5, knn, width, label='K Nearest Neighbor')
ax2.set_xticks(x, labels, rotation=45)
ax2.set_xlabel('Features')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
fig2.savefig('Features.png', bbox_inches='tight')

# RF estimators
labels = [50, 80, 100, 120, 150, 180, 200, 220, 250, 280, 300, 320, 350]
rf = [0.45, 0.417, 0.45, 0.45, 0.45, 0.483, 0.55, 0.517, 0.517, 0.517, 0.517, 0.517, 0.45]

fig3, ax3 = plt.subplots()
x = np.arange(len(labels))
width = 0.4
ax3.bar(x, rf, width, label='Random Forest')
ax3.set_xticks(x, labels, rotation=45)
ax3.set_xlabel('Number of Estimators')
ax3.set_ylabel('Accuracy (%)')
fig3.savefig('RandomForest.png', bbox_inches='tight')

# SVM
labels = ['linear', 'poly=1', 'poly=2', 'poly=3', 'poly=4', 'sigmoid', 'rbf']
svm = [0.617, 0.65, 0.1, 0.417, 0.167, 0.617, 0.583]

fig4, ax4 = plt.subplots()
x = np.arange(len(labels))
width = 0.4
ax4.bar(x, svm, width, label='Support Vector Machine')
ax4.set_xticks(x, labels, rotation=45)
ax4.set_xlabel('Kernel')
ax4.set_ylabel('Accuracy (%)')
fig4.savefig('SVM.png', bbox_inches='tight')

# KNN
labels = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6']
knn = [0.717, 0.583, 0.55, 0.35, 0.517, 0.417]

fig5, ax5 = plt.subplots()
x = np.arange(len(labels))
width = 0.4
ax5.bar(x, knn, width, label='Support Vector Machine')
ax5.set_xticks(x, labels, rotation=45)
ax5.set_xlabel('Number of Neighbors')
ax5.set_ylabel('Accuracy (%)')
fig5.savefig('KNN.png', bbox_inches='tight')

# Number of sub-samples
labels = [1, 2, 3, 4]
rf = [0.45, 0.75, 0.589, 0.673]
svm = [0.617, 0.717, 0.663, 0.7]
knn = [0.717, 0.863, 0.774, 0.782]

fig6, ax6 = plt.subplots()
x = np.arange(len(labels))
width = 0.2
ax6.bar(x - width/2, rf, width, label='Random Forest')
ax6.bar(x + width/2, svm, width, label='Support Vector Machine')
ax6.bar(x + width*1.5, knn, width, label='K Nearest Neighbor')
ax6.set_xticks(x, labels)
ax6.set_xlabel('Number of Sub-samples')
ax6.set_ylabel('Accuracy (%)')
ax6.legend()
fig6.savefig('SubsampleCount.png', bbox_inches='tight')