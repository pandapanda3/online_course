# Exploring ChatGPT parameters

## Temperature
Temperature is a parameter used to control the randomness of text generation by the model. It changes the choice of each possible word by scaling the probability distribution.

### Example

Assume the model predicts the next word with the following probability distribution:

| Word       | Probability |
|------------|--------------|
| cat        | 0.5          |
| dog        | 0.3          |
| mouse      | 0.1          |
| elephant   | 0.1          |

When `temperature < 1` (for example, `temperature = 0.5`)
The calculation steps are as follows:

1. **Scaling Probabilities**: The original probabilities are transformed to emphasize higher probabilities:

   - `cat`: \(0.5^{1/0.5} \approx 0.707\)
   - `dog`: \(0.3^{1/0.5} \approx 0.547\)
   - `mouse`: \(0.1^{1/0.5} \approx 0.316\)
   - `elephant`: \(0.1^{1/0.5} \approx 0.316\)

2. **Normalization**: Let the scaled probabilities sum to 1.

   - New probability sum: \(0.707 + 0.547 + 0.316 + 0.316 \approx 1.886\)

   After normalization:

   - `cat`: \(\frac{0.707}{1.886} \approx 0.374\)
   - `dog`: \(\frac{0.547}{1.886} \approx 0.290\)
   - `mouse`: \(\frac{0.316}{1.886} \approx 0.167\)
   - `elephant`: \(\frac{0.316}{1.886} \approx 0.167\)

The result is that the relative probabilities of higher-probability words, `cat` and `dog`, increase, while the probabilities of `mouse` and `elephant` relatively decrease.

## Maximum Length

## Top P
Top-p limits the number and range of sampled words by controlling the sum of the probability of selected words during generation.

The mechanism of top-p is to select the aggregate of the total probability and the smallest words that reach p. 

For example, if top-p=0.9, the model samples from the minimum set that brings the cumulative probability of all candidate terms to 90%. In other words, the model will first rank the candidates in order of probability from highest to lowest, then select the set of words with a probability sum of 90%, and make a random selection from this set.

### Example

Assume that the GPT model is predicting the next word in the sentence. The previous few words are "The cat is". Now, the model gives the following possible words and their probability distribution:

| Word       | Probability |
|------------|-------------|
| sleeping   | 0.5         |
| sitting    | 0.2         |
| running    | 0.1         |
| on         | 0.05        |
| eating     | 0.05        |
| playing    | 0.04        |
| jumping    | 0.03        |
| barking    | 0.02        |
| fighting   | 0.01        |

1. `top-p = 1.0`:
All candidate words are considered, and the total probability is 100%. The model can choose any word from the list.

2. `top-p = 0.9`:
We select a group of words where the cumulative probability reaches 90%. Starting from the highest probability:
- `sleeping` (0.5)
- `sitting` (0.2)
- `running` (0.1)
- `on` (0.05)

 This adds up to 0.85. To reach 0.9, we also include `eating` (0.05). Therefore, with `top-p = 0.9`, the candidates are `sleeping`, `sitting`, `running`, `on`, and `eating`, while other words (such as `playing`, `jumping`, `barking`, and `fighting`) are excluded.
## frequency penalty

A penalty that reduces the repetition of words. It is suitable for reducing redundant repetition in text.
## presence penalty

Reduces the reappearance of existing words to encourage the generation of new words.

It focuses on whether a word has already appeared in the text, regardless of the number of times the word has appeared. If a word has already appeared, the model will tend to reduce the probability of generating it again.

## best of

Generate multiple candidate texts and select the best one, suitable for scenarios that need to improve output quality.