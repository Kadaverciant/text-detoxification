# Solution Building Report

## Hypothesis 1: Vocabulary substitution

First idea that came to my mind was to create pairs of toxic words and their non-toxic substitutions. However I decided, that not only words and phrases, but also context influence on toxicity/ Therefore this approach is not good and I discarded idea to implement it.

## Hypothesis 2: Bart-paraphrase

Since detoxification task is some kind of paraphrase task, I tried to find already existing models for this, in order to tune them later. That's how I found [bart-paraphrase model](https://huggingface.co/eugenesiow/bart-paraphrase). BART has seq2seq architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT). BART is good in text generation and comprehension tasks. Our paraphrasing/detoxification task lies in this BART usage set.

## Hypothesis 3: T5-base

From NLP course I remembered about existing of Text-To-Text Transfer Transformer (T5) model. It's extremely powerful multitask text2text model, so I decided to fine-tune it for our task. During search process of fine-tuning documentation for this model I found [this kaggle notebook](https://www.kaggle.com/code/bunnyyy/t5-tuning-for-paraphrasing-questions/notebook). So all I need was to modify it for our task.

## Results

I successfully trained both models. Both BART and T5 reduced toxicity of given text. For measuring toxicity I decided to use [toxicity classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier) from given by PMLDL team [paper](https://arxiv.org/abs/2109.08914). More Information about performance of both models would be in Final Solution Report.
