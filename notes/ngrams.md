# Ngrams

Predicting future words from a given sequence of words

* Language model: statistical model of word sequences
* N-gram: Use the previous n-1 words to predict the next word

## Word counting in a corpora

### Types vs tokens
* Tokens: Total number of running words
* Types: Number of distinct words in a corpus


## Simple unsmoothed ngrams
No corpus: use uniform distribution  
Have corpus: Use P(w) = C(w)/N  
Have corupus + assume word depends on previous n-1 words: Use conditional probability
P(Wn| W1, W2, W3)

