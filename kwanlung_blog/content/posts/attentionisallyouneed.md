+++
date = '2025-04-12T17:05:50+08:00'
draft = false
title = 'Attention is All You Need'
authors = ["deanngkwanlung"]
tags = ["LLM"]
+++

## Introduction

### Why Not Use RNN?  

Mainstream approaches for text processing include RNN and encoder-decoder architectures (when structured information is abundant). However, RNNs (including LSTM and GRU) face significant limitations despite their ability to process sequential data:

Key Limitations of RNNs
1.  **High Computational Cost:**
RNNs suffer from heavy computational demands, especially on long sequences. The sequential nature of RNNs forces them to process tokens one by one (e.g., word-by-word in a sentence), making parallelization impossible. For a sequence of length $t$, RNNs require $t$ sequential steps to compute hidden states $(h_{t})$. where each $h_{t}$ depends on the previous hidden state $(h_{t-1})$ and the current input $(x_{t})$. This sequential processing leads to a time complexity of $O(t)$, making them impractical for long sequences.         
2. **Information Loss in Long Sequences:**
Historical information is compressed into fixed-size hidden states $(h_{t})$. As sequences grow longer, early-stage information tends to degrade or vanish due to the limited capacity of hidden states. This makes RNNs ineffective for capturing long-range dependencies.
3. **Gradient Issues:**
Long sequences exacerbate **gradient vanishing/explosion** problems, hindering stable training. While LSTM and GRU mitigate this to some extent, they still struggle with extremely long contexts.  
---
**The Rise of Transformer**

The Transformer architecture was introduced to address these limitations. Key advantages include:

- Parallel Computation: Unlike RNNs, Transformers process all tokens in a sequence simultaneously via self-attention, drastically reducing training time.

- Long-Range Dependency Handling: Self-attention mechanisms directly model relationships between all token pairs, regardless of distance, avoiding information decay.

- Scalability: Transformers efficiently handle long sequences, making them ideal for tasks like machine translation, text generation, and pre-trained language models (e.g., BERT, GPT).

**Attention Before Transformers**

Prior to Transformers, attention mechanisms were primarily used in encoder-decoder architectures (e.g., aligning source and target tokens in translation). However, RNN-based models still relied on sequential processing. Transformers eliminated recurrence entirely, replacing it with pure self-attention and feed-forward layers, achieving state-of-the-art performance across NLP tasks.