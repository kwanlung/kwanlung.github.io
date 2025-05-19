+++
date = '2025-04-15T00:18:56+08:00'
draft = false
title = 'Vision Transformer for Facial Expression-based PHQ-8/9 Regression'
tags = ["AI", "Deep Learning", "Computer Vision"]
+++

## Introduction
Predicting mental health scores like PHQ-8/9 from facial expressions is a challenging task that combines computer vision and affective computing. PHQ-9 (Patient Health Questionnaire-9) is a 9-item clinical survey for depression severity assessment Predicting mental health scores like PHQ-8/9 from facial expressions is a challenging task that combines computer vision and affective computing. PHQ-9 (Patient Health Questionnaire-9) is a 9-item clinical survey for depression severity assessment (PHQ-8 is similar but without the ninth item). Our goal is to build a Vision Transformer (ViT)-based model that estimates a person's PHQ-8/9 score from their facial expression. To tackle data scarcity in clinical PHQ-labeled datasets, we adopt a multi-stage training strategy:

1. **Pretrain on general emotion recognition** – First, train a ViT model to classify facial expressions on large emotion datasets (FER+, AffectNet, RAF-DB). This teaches the model to extract useful facial features

2. **Fine-tune for regression** – Next, replace the classification head with a regression head and fine-tune the model on a smaller clinical dataset with PHQ-8/9 labels, so it learns to predict a continuous score.

## Approach Overview (Multi-Stage Training)