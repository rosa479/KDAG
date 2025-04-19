# [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/pdf/1607.06520)
https://github.com/tolga-b/debiaswe

Embeddings created using a corpus naturally inherit the biases present in it. This is very dangerous because these bias will amplify themselves in the results. Job application filtering using these biased embeddings seems like a not-so-good idea. What should we do then, finding a completely non-biased dataset is impossible if we consider the fact that we will be needing a huge one for training (Word2Vec was trained with more that 10 million words). Any naturally produced text will reflect the biases present in the real world. So if there's no preventing it, let us look for the cure.

Note : There is a bit of preventing though. [Learning Gender-Neutral Word Embeddings](https://arxiv.org/pdf/1809.01496)
https://github.com/uclanlp/gn_glove
### Understanding the Bias

Word2Vec's property of word-analogies presents gender stereotypes as well.
```python 
vec = model["computer_programmer"] - model["John"] + model["Mary"]
model.most_similar([vec])[:5]
```

```
[('computer_programmer', 0.8490790724754333), ('homemaker', 0.5585830211639404), ('mechanical_engineer', 0.5584089756011963), ('electrical_engineer', 0.5357521176338196), ('schoolteacher', 0.5323439240455627)]
```
And a very childish one :
```python 
vec = model["Hot_Wheels"] - model["he"] + model["she"]
model.most_similar([vec])[:5]
```

```
[('Hot_Wheels', 0.9136512875556946), ('Barbie', 0.6337472200393677), ('Barbie_dolls', 0.5333849787712097), ('toy', 0.5218201875686646), ('dolls', 0.5054922699928284)]
```

[Rasa Algorithm Whiteboard: Measuring Bias in Word Embeddings](https://youtu.be/UwAvyACOrWs?si=-5bhDZsTKcHAqDz3)
![[Pasted image 20250227120432.png]]
This is a simplified representation of what the vectors look like in the multi-dimensional space.
It is theorized that the difference between the gender counterparts approximate the same vector as the he-she or him-her vector, while cat-dog vector doesn't.

```
Cosine Similarities Between Word Vector Differences: 
she-he vs woman-man : 0.7825 
she-he vs queen-king: 0.5988 
woman-man vs queen-king: 0.4494
```
### Quantifying the Bias

The paper used the she-he vector and assigned scores to different pairs of words from the Word2Vec embeddings
$$
S_{(a,b)}(x, y) =
\begin{cases}
\cos \left( \vec{a} - \vec{b}, \vec{x} - \vec{y} \right) & \text{if } \|\vec{x} - \vec{y}\| \leq \delta \\
0 & \text{otherwise}
\end{cases}
$$
where $\delta$ is a threshold for similarity.

The gender axis, $g$, is found and we define the direct gender bias of an embedding to be 
$$
\text{DirectBias}_c = \frac{1}{|N|} \sum_{w \in N} |\cos(\vec{w}, g)|^c
$$
where $c$ is a parameter that determines how *strict* we want to be in measuring bias. If $c = 0$, then  
$|\cos(\vec{w} - g)|^c = 0$ only if $\vec{w}$ has no overlap with $g$ and otherwise it is 1. Such strict measurement of bias might be desirable in settings such as the college admissions example from the Introduction, where it would be unacceptable for the embedding to introduce a slight preference for one candidate over another by gender. A more gradual bias would be setting $c = 1$. The presentation we have chosen favors simplicity – it would be natural to extend our definitions to weight words by frequency.
For example, in w2vNEWS, if we take $N$ to be the set of 327 occupations, then $\text{DirectBias}_1 = 0.08$, which confirms that many occupation words have substantial component along the gender direction.

The gender subspace $g$ that we have identified allows us to quantify the contribution of $g$ to the similarities between any pair of words. We can decompose a given word vector $w \in \mathbb{R}^d$ as $w = w_g + w_\perp$, where $w_g = (w \cdot g) g$ is the contribution from gender and $w_\perp = w - w_g$. Note that all the word vectors are normalized to have unit length. We define the gender component to the similarity between two word vectors $w$ and $v$ as  

$$
\beta(w, v) = \left( w \cdot v - \frac{w_\perp \cdot v_\perp}{\|w_\perp\|_2 \|v_\perp\|_2} \right) \big/ (w \cdot v).
$$

 
The intuition behind this metric is as follows: $\frac{w_\perp \cdot v_\perp}{\|w_\perp\|_2 \|v_\perp\|_2}$ is the inner product between the two vectors if we project out the gender subspace and renormalize the vectors to be of unit length. The metric quantifies how much this inner product changes (as a fraction of the original inner product value) due to this operation of removing the gender subspace. Because of noise in the data, every vector has some non-zero component $w_\perp$ and $\beta$ is well-defined. Note that $\beta(w, v) = 0$, which is reasonable since the similarity of a word to itself should not depend on gender contribution. If $w_g = 0 = v_g$, then $\beta(w, v) = 0$; and if $w_\perp = 0 = v_\perp$, then $\beta(w, v) = 1$.  

Words like $receptionist$, $waitress$ and $homemaker$ are much closer to $softball$ than they are to $football$ (which is closer to words like $businessman$ and $maestro$) with substantial $\beta$.

**Direct Bias**
- Direct bias refers to the measurable difference in how gendered words (e.g., "man" vs. "woman") are positioned in the vector space relative to other words.
- It can be quantified using the **Gender Direction**, found by computing the principal component of gendered word pairs (e.g., "he-she," "man-woman").
- Example: If "programmer" is closer to "man" in the embedding space than to "woman," this indicates direct bias.

**Indirect Bias**
- indirect bias manifests as associations between gender neutral words that are clearly arising from gender. For instance, the fact that the word receptionist is much closer to softball than football may arise from female associations with both receptionist and softball. Note that many pairs of male-biased (or female-biased) words have legitimate associations having nothing to do with gender. For instance, while the words mathematician and geometry both have a strong male bias, their similarity is justified by factors other than gender. More often than not, associations are combinations of gender and other factors that can be difficult to disentangle. Nonetheless, we can use the geometry of the word embedding to determine the degree to which those associations are based on gender.
- It is more subtle and can propagate bias across many words and concepts.
- Example: If "nursing" is closer to words related to women, while "engineering" is closer to words related to men, this reflects indirect bias.
### Eliminating the Bias

The first step in eliminating bias is to **Identify gender subspace**, or a direction which captures the embedding bias. 
For the second step we can either **Neutralise and Equalise** or **Soften**. 

[Rasa Algorithm Whiteboard - Using Projections to Remove Bias from Word Embeddings](https://youtu.be/8xQbWlCEHRw?si=dkjjAtCIw11GIMOF)
![[Pasted image 20250227122224.png]]

Neutralise ensures that gender neutral words are zero in the gender subspace. Equalise ensures that any neutral words are equidistant to all words in each equality set.
For instance, if $\{grandmother, grandfather\}$ and $\{guy, gal\}$ were two equality sets, then after equalization $babysit$ would be equidistant to grandmother and grandfather and also equidistant to gal and guy, but presumably closer to the grandparents and further from the gal and guy. This is suitable for applications where one does not want any such pair to display any bias with respect to neutral words.
#### **Steps in Hard Debiasing:**

1. **Identify the Gender Subspace**
    
    - Compute the **gender direction** in the embedding space by taking the principal component of word pairs like **(he-she, man-woman, king-queen)**.
    - This direction captures gender-related differences.
2. **Neutralize Gender for Neutral Words**
    
    - Identify gender-neutral words (e.g., "doctor," "nurse," "programmer").
    - Project these words onto the gender subspace and then remove that component, making them equidistant from "male" and "female" words.
3. **Equalize Gender Pairs**
    
    - Ensure that words with natural gender pairs (e.g., **"actor-actress"**, **"king-queen"**) are equidistant from gender-neutral words.
    - This prevents bias removal from distorting valid semantic relationships.


Instead of completely removing gender components, soft debiasing **reduces** the strength of bias while preserving useful information. It is often used in methods like **regularization, fine-tuning, and re-weighting loss functions** during training.
#### **Techniques in Soft Debiasing:**

- **Loss Function Regularization:** Modify the model’s training process to discourage gendered associations.
- **Counterfactual Data Augmentation:** Introduce gender-balanced training examples (e.g., replacing "he" with "she" in some sentences).
- **Post-processing Adjustments:** Shift word vectors slightly rather than completely removing gender information.
### Discussion
There exist racial, ethnic and cultural stereotypes within the Word2Vec database as well to which we can apply similar methods.
Here we work along a male-female axis. Can we reduce gender identity to a straight line? What about other languages like German or Sanskrit where there are three classifications i.e. male, female, neutral?
We have to be really careful about application and use-case and not give free reign to the model working off the embeddings.
### Further Reading
[‘Thy algorithm shalt not bear false witness’: An Evaluation of Multiclass Debiasing Methods on Word Embeddings](https://arxiv.org/pdf/2010.16228)
### [Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them](https://arxiv.org/pdf/1903.03862)
- However, as we show in this work, while the gender-direction is a great indicator of bias, it is only an indicator and not the complete manifestation of this bias. We claim that the bias is much more profound and systematic, and that simply reducing the projection of words on a gender direction is insufficient: it merely hides the bias, which is still reflected in similarities between “gender-neutral” words.

![[Pasted image 20250228081108.png|400]]

- Can a classifier learn to generalize from some gendered words to others based only on their representations? We consider the 5,000 most biased words according to the original bias (2,500 from each gender), train an RBF-kernel SVM classifier on a random sample of 1,000 of them (500 from each gender) to predict the gender, and evaluate its generalization on the remaining 4,000. For the HARD-DEBIASED embedding, we get an accuracy of 88.88%, compared to an accuracy of 98.25% with the non-debiased version. For the GN-GLOVE embedding, we get an accuracy of 96.53%, compared to an accuracy of 98.65% with the non-debiased version.