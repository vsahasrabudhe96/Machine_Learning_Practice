# Naive Bayes

### Here we implemented Gaussian NB and Multinominal NB

- Multinomial
```
Similarly as before, we notice that the more dollar signs ($) there are in an email, the more likely that email is spam. We can do this for many kinds of words, say (CASH or Lottery), but instead of labeling them 0 or 1, we actually count how many times each word appears in the email. This helps the model by giving it information, not just on whether the word was there, but also how many times the word appeared because we know that this is a signal to help our classifier. The algorithm assumes that the features are drawn from a multinomial distribution.
```

For Gaussian, let’s assume we’re trying to classify whether a college student can dunk a basketball based only on their height.

- Gaussian
```
As you may recall from any intro stats class, the distribution of heights in humans is continuous and normally distributed (the normal distribution is also called a Gaussian distribution, hence the name). So the algorithm will look at the height of all of the students we polled and determine where the cut-off should be to maximize the model performance (usually accuracy) to classify dunkers vs non-dunkers.
```


Reference : https://www.quora.com/What-is-the-difference-between-the-the-Gaussian-Bernoulli-Multinomial-and-the-regular-Naive-Bayes-algorithms