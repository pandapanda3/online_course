# LIME (Local Interpretable Model-Agnostic Explanations) Notes

## Overview
- **Introduced**: 2016.
- **Purpose**: Explains individual predictions of black-box machine learning models using interpretable models.
- **Advantages**:
  - Works with **tabular data**, **text**, and **images**.
  - Highlights the features most contributing to a prediction, aiding trust in the model.


## How LIME Works
1. **Select an instance of interest** to explain.
2. **Perturb the dataset** and obtain predictions for the perturbed samples.
3. **Create a new dataset** with the perturbed samples and their predictions.
4. **Train an interpretable model** (e.g., linear regression) weighted by proximity to the instance of interest.

### Key Points
- **Local explanation**: Focused on one instance or prediction, not generalizable globally.
- **Loss function**: Minimizes the difference between the explanation and the black-box prediction.
- **Model complexity**: Kept low (fewer features) based on Occam's Razor.


## Applications
### 1. Text
- Variations are created by randomly **removing words**.
- Example: In spam classification, removing "channel" strongly impacts the spam prediction.

### 2. Images
- **Segment images** into superpixels and replace them with a defined color (e.g., gray).
- Example: Identifying top features for image classification like guitars or huskies.

### 3. Tabular Data
- Highlights feature importance (e.g., temperature or weather for bike rental predictions).
- **Limitation**: Sampling may not occur near the instance of interest but from the dataset's mass center.


## Challenges with LIME
- **Neighborhood definition**: Difficult to optimize proximity for local models.
- **Kernel width**: Impacts influence of data points; hard to tune in high-dimensional spaces.
- **Consistency issues**: Results may vary across different settings or experiments.
- **Risk of bias**: Can obscure biases and fool users (Slack et al., 2020).



## Pros and Cons
### Pros
- **Flexible**: Model-agnostic, works with any machine learning model.
- **Intuitive**: Easy to understand and explain predictions.
- **Supports diverse data types**: Text, images, tabular data.

### Cons
- **Neighborhood sampling**: Imprecise, especially for tabular data.
- **Consistency**: Results may differ under different conditions.
- **Ethical risks**: Potential misuse to hide biases or manipulate interpretations.



## Practical Notes
- **Kernel smoothing**: Uses an exponential kernel to weigh data points.
- **Model simplicity**: Users determine the maximum number of features in the linear regression.
- **Occam’s Razor**: Prefer simpler, more interpretable explanations when competing hypotheses exist.



# Notes on Anchors for Explainable AI

## Overview
- **Introduced By**: Authors of the original LIME paper.
- **Purpose**: Explain individual predictions using **if-then rules** that "anchor" predictions, ensuring high precision.
- **Motivation**: Addresses limitations of LIME by avoiding linear approximation of potentially nonlinear or complex decision boundaries.

## Key Concepts
1. **Anchors**: Decision rules that anchor predictions for an instance of interest.
   - Rule precision: Evaluates neighbors of the instance matching the anchor rule.
   - Coverage: Probability that the anchor applies to its neighbors.
   - Precision Threshold (Tau): Minimum precision required for a rule to be valid.
2. **Trade-Off**: Higher coverage may reduce precision and vice versa.

## Components of Anchor Explanations
1. **Candidate Generation**: Generates initial explanation candidates (one per feature of the instance) and extends them in subsequent rounds.
2. **Best Candidate Identification**: Uses a multi-armed bandit algorithm to compare rules, minimizing model calls while optimizing precision.
3. **Candidate Precision Validation**: Ensures statistical confidence in rules surpassing the precision threshold.
4. **Modified Beam Search**: Retains the top B candidates per round and creates new rules by adding predicates, balancing computational load and performance.

## Examples in Practice
- **Tabular Data**: Simple **if-then** anchors explain predictions (e.g., predicting health outcomes).
- **Images**: Superpixels identify areas contributing to predictions (e.g., anchor for "beagle").
- **Machine Translation**: Highlights key words influencing translation predictions.

## Challenges with Unbalanced Data
- Perturbation spaces reflect training data distribution, leading to:
  - **Empty Rules** for majority class.
  - **Overly Specific Rules** for minority class.
- **Mitigation Strategies**:
  - Define custom perturbation spaces.
  - Use a subset of balanced training data.
  - Adjust bandit parameters to sample minority instances more often.

## Pros
- **Human-Understandable**: Clear, intuitive if-then rules.
- **Model-Agnostic**: Works with any machine learning model.
- **Handles Nonlinear Predictions**: Effective for locally complex boundaries.

## Cons
- **Parameter Tuning**: Requires careful adjustment for optimal performance.
- **Perturbation Design**: Must be tailored for each use case.
- **Coverage Limitations**: Undefined in many domains, complicating comparisons.

## Summary
Anchors provide a highly interpretable approach to explainable AI, particularly useful for classification tasks with nonlinear decision boundaries. However, careful design of parameters and perturbations is essential for effective application.

# Shapley Values

## Overview
- **Introduced**: 1952 as a concept in coalitional game theory.
- **Purpose**: Fairly distribute rewards (e.g., model predictions) among contributors (e.g., features) based on their contributions.
- **Analogy**: In a hackathon, Shapley values help divide the prize among team members based on individual and collaborative contributions.

## How It Works
1. **Coalition Analogy**:
   - Features in a model = Team members in a coalition.
   - Model prediction = Payout.
   - Shapley values = Contribution of each feature to the prediction.
2. **Contribution Analysis**:
   - Evaluate individual contributions.
   - Consider interaction effects between features (or team members).
3. **Subset Averaging**:
   - Use all possible subsets of features to calculate each feature’s marginal contribution.
   - Average contributions across subsets to derive the Shapley value.

## Interpretation
- **Correct Interpretation**:
  - The Shapley value reflects a feature’s contribution to the difference between the actual prediction and the mean prediction.
- **Incorrect Interpretation**:
  - It is **not** the difference in predictions after removing a feature from the model.

## Mathematical Approach
- **Key Steps**:
  1. Compare predictions with and without the feature in subsets.
  2. Weight contributions by the size of the subset.
  3. Sum contributions across all subsets.
- **Practical Implementation**:
  - Replace features with random values from the training data instead of fully removing them.

## Pros and Cons
### Pros:
- **Fair Distribution**: Accurately distributes contributions among features.
- **Contrastive Explanations**: Enables comparisons with mean predictions or subsets.
- **Theoretical Foundation**: Not based on assumptions (e.g., local linearity like LIME).

### Cons:
- **High Computational Cost**: Requires \(2^n\) subset evaluations, making it computationally expensive for large feature sets.
- **Requires Approximation**: Practical implementations need approximation methods.
- **Risk of Misinterpretation**: Can be misunderstood if not applied correctly.
- **Feature Dependency**: Requires all features and access to data for computation.

## Key Takeaways
- Shapley values provide a rigorous, theoretically grounded way to interpret feature contributions in machine learning models.
- Despite their utility, they require careful handling due to their computational complexity and potential for misinterpretation.



## A Simple Example
Imagine a team of 3 members (A, B, C) who worked together and earned $100.  
We want to fairly distribute the $100 based on each member's contribution.

### Contributions:
- **A alone**: $60.
- **B alone**: $20.
- **C alone**: $10.
- **A + B**: $80.
- **A + C**: $70.
- **B + C**: $40.
- **A + B + C**: $100.

## Step 1: Calculate Marginal Contributions
For each member, compute their marginal contribution in all possible team combinations:

| Team Combination  | Marginal Contribution of A | Marginal Contribution of B | Marginal Contribution of C |
|--------------------|----------------------------|----------------------------|----------------------------|
| { → {A}          | 60                         | -                          | -                          |
| {} → {B}          | -                          | 20                         | -                          |
| {} → {C}          | -                          | -                          | 10                         |
| {B} → {A, B}      | 80 - 20 = 60               | -                          | -                          |
| {C} → {A, C}      | 70 - 10 = 60               | -                          | -                          |
| {A} → {A, B}      | -                          | 80 - 60 = 20               | -                          |
| {A} → {A, C}      | -                          | -                          | 70 - 60 = 10               |
| {B} → {B, C}      | -                          | 40 - 20 = 20               | -                          |
| {C} → {B, C}      | -                          | -                          | 40 - 20 = 20               |
| {A, B} → {A, B, C}| 100 - 80 = 20              | 100 - 70 = 30              | 100 - 80 = 20              |

## Step 2: Average Marginal Contributions
Shapley values are the **average of all marginal contributions** for each member.

### A's Shapley Value:
A = (60 + 60 + 60 + 20)/4 = 50

### B's Shapley Value:
B = (20 + 20 + 30)/4 = 25


### C's Shapley Value:
C = (10 + 20 + 20)/4 = 25


## Step 3: Final Distribution
- **A**: $50.
- **B**: $25.
- **C**: $25.

### Summary
The Shapley values ensure the $100 is distributed fairly based on individual and collaborative contributions.

# SHAP (SHapley Additive exPlanations)

## Overview
- **Introduced**: 2017, builds on the concept of Shapley values.
- **Purpose**: Approximates Shapley values to reduce computational cost.
- **Methods**:
  - **Kernel SHAP**: Similar to LIME, uses linear regression to approximate Shapley values.
  - **Tree SHAP**: Specialized for tree-based models.
  - **Deep SHAP**: Designed for deep learning models.


## Kernel SHAP: Steps
1. **Sampling Coalitions**: Generate feature subsets.
2. **Prediction**: Convert subsets to original feature space and apply the model to get predictions.
3. **Weighting**: Compute weights for each sample using the SHAP kernel.
4. **Linear Regression**: Fit a weighted linear regression model.
5. **Output**: Coefficients of the model approximate Shapley values.


## Advantages
- **Fair Distribution**: Distributes prediction difference from the average prediction among features.
- **Contrastive Explanations**: Enables comparison to the average prediction, subsets, or individual data points.
- **Faster than Shapley Values**: Approximations significantly reduce computational cost.
- **Visualization**: Produces clear and interpretable feature attributions, often outperforming methods like LIME.


## Limitations
- **Computational Cost**: Kernel SHAP, while faster, can still be slow for large datasets.
- **Feature Independence Assumption**: Assumes features are independent, which may not always hold.
- **Data Dependency**: Requires access to data to compute attributions.
- **Potential for Misuse**: Can create misleading interpretations or hide biases intentionally.


## Summary
SHAP provides a more practical and interpretable method for feature attribution by approximating Shapley values. While it reduces computational overhead compared to exact Shapley values, it still has limitations like potential misuse and reliance on feature independence.
### Key Differences Between SHAP and Shapley Values

1. **Purpose**: Shapley values are a theoretical concept from game theory, while SHAP adapts this concept for machine learning model explanation.
2. **Efficiency**: Shapley values require exhaustive computation of all feature subsets (\(2^n\)), making them impractical for large models. SHAP uses approximation methods (e.g., Kernel SHAP, Tree SHAP) to reduce computational cost.
3. **Model Application**: Shapley values are general-purpose, while SHAP is specifically designed to explain feature contributions in machine learning.
4. **Usability**: SHAP provides optimized algorithms for different models (e.g., tree-based or deep learning) and includes intuitive visualizations, whereas Shapley values lack such practical enhancements.
# Summary of ICE Plots (Individual Conditional Expectation Plots)

## Overview
- **Introduced**: 2014 by Goldstein et al.
- **Purpose**: Visualize how individual predictions change as a single feature varies, improving local explainability and global understanding.

## How ICE Plots Work
1. **Select Instance and Feature**: Choose a single data point (instance) and a feature of interest.
2. **Keep Other Features Constant**: Create variants of the instance by changing the selected feature while holding other features fixed.
3. **Predict Values**: Use the model to predict outcomes for each variant.
4. **Plot Results**: Plot a line for each instance, showing feature values (x-axis) vs. predictions (y-axis).

## Variations
1. **Centered ICE Plots (c-ICE)**:
   - Center curves at a specific feature value to show relative changes in predictions.
   - Highlights trends more clearly by eliminating differences in starting predictions.
2. **Derivative ICE Plots (d-ICE)**:
   - Show derivatives of predictions with respect to the feature.
   - Helps identify ranges of feature values where predictions change significantly.

## Example
We have a machine learning model that predicts **bicycle rentals** based on the following features:
1. **Temperature** (°C)
2. **Weather condition** (e.g., sunny, rainy)
3. **Wind speed** (km/h)

We want to understand how **temperature** affects the number of bicycle rentals while keeping other features constant.

### Steps:
1. **Choose an Instance**:
   - **Weather condition**: Sunny
   - **Wind speed**: 5 km/h
   - **Temperature**: 20°C
   - **Prediction**: 500 rentals

2. **Vary the Temperature**:
   - Keep "Weather condition" and "Wind speed" fixed.
   - Change the temperature: 0°C, 10°C, 20°C, 30°C, 40°C.

3. **Get Predictions**:
   - 0°C → **200 rentals**
   - 10°C → **400 rentals**
   - 20°C → **500 rentals**
   - 30°C → **450 rentals**
   - 40°C → **300 rentals**

4. **Plot**:
   - X-axis: Temperature (°C)
   - Y-axis: Predicted rentals
   - The resulting curve shows the effect of temperature on predictions for this instance.


### Multiple Instances
- Repeat the process for other samples in the dataset.
- Each sample generates a unique curve, showing temperature's effect across different conditions.


### Observations
1. Some instances show increasing rentals with higher temperatures.
2. Other instances show rentals dropping after a peak temperature (e.g., 30°C).
3. ICE plots reveal how predictions vary across samples, offering **local and global interpretability**.

## Pros and Cons of ICE Plots
### Pros:
- **Intuitive**: Easy to interpret for a single feature.
- **Local and Global Insights**: Uncover heterogeneous relationships and individual trends.
- **Customizable**: Variations like c-ICE and d-ICE enhance clarity.

### Cons:
- **Single Feature Only**: Cannot display relationships involving multiple features.
- **Overcrowding**: Too many instances can make plots cluttered.
- **Correlation Issues**: If the selected feature is correlated with others, generated data points might not reflect realistic scenarios.

## Applications
- Useful for understanding how specific features impact predictions in black-box models.
- Can highlight trends and anomalies at both local (individual instance) and global (overall dataset) levels.


# Functional Decomposition in Explainable AI (xAI)

## Overview
- **Definition**: Functional decomposition divides complex machine learning models into simpler parts (main effects, interaction effects, and intercept).
- **Purpose**: To make it easier to analyze and understand the overall behavior of the model by focusing on individual components.


## Components of Functional Decomposition
1. **Main Effects**:
   - How each feature affects the prediction independently of other features.
2. **Interaction Effects**:
   - The joint effect of multiple features on the prediction.
3. **Intercept**:
   - The baseline prediction when all feature effects are set to zero.


## Methods for Functional Decomposition
1. **Functional ANOVA**:
   - **Additive Decomposition**:
     - Decomposes the prediction function \(f(x)\) into a sum of terms, where each term represents the contribution of one or more features.
   - **Individual Contributions**:
     - Identifies how much each feature and interaction contributes to the model's output.
   - **Quantifying Interaction Effects**:
     - Measures the combined effect of multiple features on predictions.
2. **Accumulated Local Effects (ALE)**:
   - A method to analyze main and interaction effects (covered later).


## Visualization Example

### Scenario
We have a machine learning model predicting **house prices** based on:
1. **Area** (in square meters)
2. **Location** (City or Suburb)
3. **Renovation** (Basic or Premium)

For a specific house:
- **Area**: 150 m²
- **Location**: City
- **Renovation**: Premium
- **Prediction**: **500,000 USD**

### Functional Decomposition
1. **Intercept**: Base price (no feature effects): **150,000 USD**.
2. **Main Effects**:
   - Area: **200,000 USD**
   - Location (City): **100,000 USD**
   - Renovation (Premium): **50,000 USD**
3. **Interaction Effects**:
   - Area × Location: **50,000 USD**

### Final Prediction:
Prediction = Intercept + Main Effects + Interaction Effects

500,000 = 150,000 + (200,000 + 100,000 + 50,000) + 50,000


### Visualization:
- **Main Effects**: Plot individual contributions of Area, Location, and Renovation.
- **Interaction Effects**: Use a 3D plot or heatmap to show Area × Location interaction.

This example highlights how functional decomposition clarifies the impact of features and their interactions on predictions.


## Pros and Cons of Functional Decomposition
### Pros:
1. **Global Explanations**:
   - Breaks down model behavior into understandable components, supporting global interpretability.
2. **Theoretical Foundation**:
   - Provides a justified framework for analyzing high-dimensional models.
3. **Feature Contributions**:
   - Clearly quantifies the effects of individual features and their interactions.

### Cons:
1. **Computational Complexity**:
   - Decomposing models can be computationally intensive.
2. **Limited Applicability**:
   - Best suited for **tabular data**; less effective for image or text data.


## Summary
Functional decomposition simplifies complex models by dividing predictions into main effects, interaction effects, and intercepts. It supports global explanations and helps quantify feature contributions but can be computationally demanding and is primarily suitable for tabular data.
# Feature Interaction in Explainable AI

## Overview
Feature interaction examines how multiple features jointly influence model predictions. In functional decomposition, predictions are broken into:
1. **Constant term**: Base prediction (e.g., $30,000 for gasoline non-luxury cars).
2. **Feature terms**: Individual contributions (e.g., +$20,000 for electric fuel, +$40,000 for luxury brand).
3. **Interaction terms**: Combined effects of features (e.g., fuel type and brand interacting).


## Example

We have a machine learning model predicting **car prices** based on:
1. **Fuel Type**: Gasoline or Electric.
2. **Brand Type**: Non-Luxury or Luxury.

We want to analyze whether there is a significant **interaction** between **Fuel Type** and **Brand Type** using the **H statistic**.

### Step 1: Model Predictions
The model’s predictions for different feature combinations are:
| Fuel Type  | Brand Type   | Predicted Price (USD) |
|------------|--------------|-----------------------|
| Gasoline   | Non-Luxury   | 30,000               |
| Gasoline   | Luxury       | 70,000               |
| Electric   | Non-Luxury   | 50,000               |
| Electric   | Luxury       | 120,000              |

### Step 2: Interaction Analysis
1. **Main Effects**:
   - **Fuel Type**:
     - Gasoline → $0 (baseline)
     - Electric → +$20,000
   - **Brand Type**:
     - Non-Luxury → $0 (baseline)
     - Luxury → +$40,000
2. **Expected Prediction Without Interaction**:
   - For **Electric Luxury**:
     Expected Price = Base Price + Fuel Effect + Brand Effect
     
     Expected Price = 30,000 + 20,000 + 40,000 = 90,000 USD.
     
3. **Observed Prediction**:
   - Electric Luxury → $120,000.
4. **Interaction Contribution**:
   Interaction = Observed - Expected = 120,000 - 90,000 = 30,000 USD.
   

### Step 3: Quantifying Interaction with H Statistic
The **H statistic** quantifies how much of the variance in predictions is caused by the interaction:

H = Variance due to Interaction/Total Variance in Predictions

- If (H = 0: No interaction exists.
- If (H > 0): Interaction contributes significantly to predictions.
- For this example, a high (H) indicates a strong interaction between **Fuel Type** and **Brand Type**.

---

### Key Takeaways
- The interaction between **Fuel Type** and **Brand Type** contributes $30,000 to the price of Electric Luxury cars.
- The **H statistic** provides a numerical measure of interaction strength.
- This analysis helps identify how combined features influence predictions beyond their individual effects.


## Measuring Interaction: H Statistic
### Introduced by Friedman and Popescu (2008)
1. **Two-Way H Statistic**:
   - Measures interaction strength between two features j and k.
2. **Total H Statistic**:
   - Measures interaction strength between a feature j and all other features.

### Key Metrics:
- **H = 0**: No interaction.
- **H = 1**: All variation explained by feature interaction.
- **Partial Dependence**:
   - Iterates over all data points, evaluating the joint effect of features.



## Challenges with the H Statistic
1. **Computational Cost**:
   - Worst-case complexity: 2n^2 for two-way interactions, 3n^2 for total interactions.
   - **Solution**: Sample data points to reduce cost, but this increases variance.
2. **Interpretation**:
   - If H > 1, results can be difficult to explain.
   - High variance in estimates can reduce stability.



## Pros and Cons of Feature Interaction Analysis
### Pros:
1. **Dimensionless**: Allows comparisons across features and models.
2. **Meaningful Insights**: Highlights joint effects of features on predictions.

### Cons:
1. **Computationally Expensive**: Requires many calls to the model's predict function.
2. **Unstable Results**: Sampling increases variance in estimates.
3. **Correlation Issues**: Ineffective for highly correlated features.
4. **Interpretability**: H > 1 can make results challenging to understand.



## Conclusion
Feature interaction analysis, particularly with the H statistic, provides valuable insights into how features jointly affect predictions. However, practical challenges such as computational cost, instability, and interpretability must be carefully managed.
