+++
date = '2025-04-12T17:05:50+08:00'
draft = false
title = 'Attention Is All You Need: A Beginner-Friendly Guide to Transformers'
authors = ["deanngkwanlung"]
tags = ["LLM"]
+++

## Introduction

In 2017, the paper *“Attention Is All You Need”* introduced a new neural network architecture called the Transformer that revolutionized natural language processing. Transformers moved away from the traditional recurrent neural networks (RNNs) and instead relied entirely on a mechanism called attention to handle sequences of data. This breakthrough enabled models that are faster to train, better at capturing long-range dependencies in text, and ultimately led to modern language models like BERT and GPT. In this post, we will build an intuitive understanding of the Transformer architecture. We’ll start by examining why earlier sequence models (like RNNs) struggled, then explain attention mechanisms (especially self-attention and multi-head attention), and finally walk through the Transformer’s encoder-decoder architecture including positional encoding. By the end, you should understand how and why Transformers work so well – in clear, beginner-friendly terms.

## Background: RNNs and Their Limitations

**Recurrent Neural Networks (RNNs)** were long the dominant approach for sequence tasks (such as language modeling and translation). In an RNN, each input token (word) is processed in sequence, updating a hidden state that carries information from previous tokens. Variants like LSTMs and GRUs improved vanilla RNNs by mitigating some training difficulties. However, RNN-based models still face significant limitations when dealing with long sequences or complex language:

- **High Computational Cost (No Parallelization)**: RNNs process tokens one by one in sequence. To process a sequence of length $t$, an RNN must perform $t$ sequential updates – you can’t process the $t$-th token until you’ve processed the $t-1$ tokens before it. This inherent sequential operation (each hidden state $h_t$ depending on $h_{t-1}$) prevents parallel computation. Even with optimized implementations, it’s hard to fully utilize modern hardware since you can’t break the dependency chain. The sequential nature leads to a time complexity of $O(t)$ per sequence, which becomes impractical for very long sequences.
- **Difficulty Handling Long-Range Dependencies**: As sequences grow longer, RNNs struggle to retain information from far-back tokens. The entire history must be compressed into the fixed-size hidden state. Early information can gradually fade out as it gets overwritten by more recent inputs. Although LSTMs/GRUs introduce gating mechanisms to improve long-term memory, in practice there is still a limit to how well very distant dependencies can be preserved in a single vector. Important context from the beginning of a paragraph might be lost by the time the RNN reaches the end.
- **Gradient Vanishing/Explosion**: RNNs trained on long sequences are notorious for unstable gradients. The backpropagated gradients either shrink (vanish) or grow (explode) through the long chain of multiplications through time, making it difficult to learn parameters effectively. Gated RNNs alleviate this to some extent but do not fully solve it for extremely long sequences.

These issues meant that RNNs could be *slow* to train, and even once trained they might fail to capture long-term patterns in text. Researchers tried to address the bottlenecks – for example, using **convolutional neural networks (CNNs)** to process sequences in parallel. CNNs can indeed operate on sequence segments simultaneously, reducing the sequential depth, but they have a limited receptive field. A convolution with a small kernel can’t directly connect tokens far apart; it would require many convolutional layers to cover long distances. This makes very long-range dependency modeling still challenging with pure CNNs, and the computation cost is shifted to having many layers or large kernels. In short, CNN-based sequence models didn’t fully solve the problem either.

**The Rise of the Transformer – Parallelism and Long-Range Attention**: The Transformer architecture was introduced to directly tackle these issues. Instead of processing one token at a time, Transformers leverage self-attention to look at all tokens in a sequence in parallel. This yields several key advantages:

- **Parallel Computation**: A Transformer does not have to iterate through each position sequentially. **Self-attention** allows each token to attend to others in one big operation, so we can process entire sequences simultaneously. This drastically reduces training time compared to RNNs, especially on modern hardware where matrix operations can be highly parallelized.
- **Long-Range Dependency Handling**: In self-attention, every token can directly “see” every other token, regardless of their distance in the sequence. The attention mechanism computes interactions (similarity) between all pairs of tokens. This means even tokens far apart can influence each other’s representations in a single attention layer, avoiding the problem of information fading over long distances.
- **Better Gradient Flow**: With parallel attention and shorter paths between tokens, gradient propagation is more direct. Additionally, the Transformer uses techniques like residual connections and layer normalization (more on these later) that help maintain stable gradients in deep networks.
- **Scalability**: In practice, Transformers scale very well with data and model size. We can train very deep Transformers on huge datasets by leveraging the parallelism. This has led to the era of large-scale pre-trained language models (GPT, BERT, etc.) that achieve state-of-the-art results on numerous tasks. Transformers have proven effective not just in NLP but also in speech, vision, and other domains, due to their flexible architecture.

**Attention Before Transformers**: It’s worth noting that attention mechanisms were not entirely new – they existed as add-ons to RNN encoder-decoder models for tasks like machine translation prior to the Transformer. In those models, an encoder RNN would produce a sequence of hidden states, and a decoder RNN would “attend” to the encoder outputs to decide which parts of the input are relevant to the current output step. This was introduced by Bahdanau et al. (2015) and significantly improved translation quality by allowing the model to focus on relevant words instead of relying on a single fixed context vector. However, those models still used **RNNs** at their core, processing word by word sequentially. The Transformer’s innovation was to eliminate recurrence entirely – it relies solely on attention (and feed-forward networks) for both encoding and decoding, making it the first architecture to dispense with RNNs in sequence modeling.  
With this background in mind, let’s dive into what attention actually is and how it works, since it’s the fundamental ingredient in Transformers.

## Attention Mechanisms Explained

At a high level, an **attention mechanism** lets a model **dynamically weight** the importance of different pieces of information. Rather than processing all inputs with equal emphasis, the model learns to focus on the parts that are most relevant to the task at hand. In the context of sequences (like sentences), this means at any given time step or for any given word, the model can attend to (i.e. look at) other specific words in the sequence that help inform its output.

### Key-Value Attention Concept

The general form of attention can be described in terms of **queries, keys, and values** (often abbreviated as Q, K, V). You can think of this process like a search query in information retrieval:

- The **query** is what we are looking for – e.g. the word we are trying to process or a representation of the current context.
- The **keys** are indices or tags for our knowledge base – e.g. representations of all words in the sequence (or in another sequence, if it’s cross-attention).
- The **values** are the actual information we can retrieve – e.g. the representations of those words that will be combined to produce some result.

The **attention mechanism** will compare the query against each key to compute a similarity score (often called an *alignment score*). These scores determine how much weight to give to each corresponding value. Finally, the attention output is a weighted sum of the values, weighted by those similarity scores (after normalization).

Concretely, imagine a simple example: we have a sentence, and we want to encode a particular word with attention. The query could be the embedding of that word, the keys could be embeddings of all words in the sentence, and the values could also be the embeddings of all words (in basic self-attention, keys and values come from the same source). If the query word is, say, a pronoun, the attention mechanism can learn to assign higher weight to the key (and value) of the noun that the pronoun refers to, effectively allowing the model to gather information from that relevant noun to better represent the pronoun.