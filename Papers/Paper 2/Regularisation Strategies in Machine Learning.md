# [A comprehensive survey on regularization strategies in machine learning](https://pdf.sciencedirectassets.com/272144/1-s2.0-S1566253521X00115/1-s2.0-S156625352100230X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFQaCXVzLWVhc3QtMSJGMEQCIA%2BKvSWvAUR5Elj4kYPSL8LoxSrHb4mq%2Bq2K5N%2BcGBZzAiAXgslL9Gob73TB2BxFT7giISMIODqFoEwoU401ILaSCCq7BQiN%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMxgRcs84IdWq%2BV6EaKo8Flg8%2FCKLDEycMJoshDHa%2BaWaD5o07Gi4PfebDc54GWRnFf15JI10nX%2FqXjRXnV2E5GU0uyOLbGkqjNzxSAKcvMonqyzxXJM%2BqPK4hH26%2FdIy%2FXyym0vIn251wHtcwmQ6Jf4YKHoFiE3p2zwpG%2BdZjF6jHE4tVB5wOfiE%2F4Go4T9LZOT4xCfG9C7fnnjkOeOiHQF6vBMSM%2FY6uAzvRBjtF2M8c%2FRjRH6Mty3a2bHemoHKm5a3uwPKYPfYnwXEswLNetoF1OeNCF5MC%2B7bF6b2jtkh1JyAxLmct%2B3zJBnT4Ty55mtM9CidwXprKLwWKrp0EQRB5G8njt3P4kFZ%2BDeNgD5J06bcA22UTWSWc4vztVgENxVNXyTKYSUeVchmCxfLNNWrfvLj9nGXgLZr76tQvyJeLzj2isKySQqkefrSCOYrYQ9AIrLteqf775lmCt%2FKdB8y2e6vfscHS3raCCqnKebDA4GqMpPZXfIu9UMOOU9Ae0LtwUnw2IURgl18TRC5LKYa3o2e%2BlG%2FuYMHEaczj51ve8c0kkk9LZy24QZ39EwmTK9CT3S%2FmpdYSmXWgppZ3oG%2B82WQ8iHBV2SkfLFcUaQ9rPkKaUyMKU6FcPTVlLskhtIiTQ1ONcb72iOKV%2FCIDyjq8%2Fr%2FAOodGUvjR9jHHEF4nbpav%2Bxd%2B2zRO0rWjr60Usl1C06dkTzYCwpzcp77DxUwGB824QwF1KIGFBaEEJuJ3MTVDAy9qDRVhugPAsMg5z%2B4QfKXp1EbICAD577Kscg8Tq%2Fbq6q3aHGQG0i%2BXBIf32c08Jzvs%2FJm4HBOaWPx2qsIkeX97hA6YeT2iwMeLIWFdm9C2zN4%2FdZFJdTS%2FpO%2BtGFrbpxyh30zNIRsd%2BDDFu4a%2BBjqyAbfjhADaQL4S4TRy99Q324btt7LmeAJpesV908vmXb%2B7VHQb6dLAlMdNoKK7if0DNc3Su%2Fxf9WkabgSDaBZTibcEVPhJ4oDbUmot6KigzX5gtpNM7QhE%2FAt1tEgsS4SwwlEHTbqP1AUoWBP9fFclFkKB88%2BuOOtE6wr9o%2F2bXz3DJr3JMxeM7yX4OXHyTlsDKK56ebpSDeUdqHaycyDraSUlTcl%2FS8AvpXTxbRPctg%2F9YMs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250228T123739Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWG6YYSDF%2F20250228%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c42efa99270695f3cfe27b220372529f5297cae31ae6285407f0a7d54330531f&hash=e63a56af3d4656211ebe1bdc93acb3ee95d5142c34b4a6df4645bb81f57a2488&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S156625352100230X&tid=spdf-9def0f61-16fa-4051-a4dd-d64319b4bbcf&sid=9deb20ab14e2e04f8259456-46b32d7b15d0gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0c085d5356515050595f&rr=919077989c218afd&cc=in)

The concept of overfitting is that the loss decreases on training data, but the loss increases on unseen data.

![[Pasted image 20250228221857.png|500]]

Given limited data, the model may *overlearn* the training data and cause issues.
Regularisation remedies this problem by giving good generalisations to unseen data even when trained on less data and in presence of noise.

The loss function for linear regression (Mean Squared Error - MSE) is given by: $$ L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 $$ where: 
- $m$ is the number of training examples, 
- $y_i$ is the actual value, 
- $\hat{y}_i$ is the predicted value.
# Common regularisation techniques for linear models are :

**Lasso Regression** - (L1 Regularization)
	L1 regularization modifies the loss function as: $$ L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} |\theta_j| $$ L1 regularization promotes sparsity by shrinking some parameters to zero.

**Ridge Regression** - (L2 Regularisation)
	L2 regularization adds a penalty term to the loss function: $$ L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} \theta_j^2 $$ where: - $\lambda$ is the regularization parameter, - $\theta_j$ are the model parameters. L2 regularization prevents overfitting by penalizing large weights.

**Elastic Net Regression** - (L1 and L2 Regularisation)
	Elastic Net regularization combines L1 (Lasso) and L2 (Ridge) penalties: $$ L(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2 $$ where: - $\lambda_1$ controls the L1 penalty (sparsity), - $\lambda_2$ controls the L2 penalty (weight shrinkage), - $\theta_j$ are the model parameters. Elastic Net is useful when there are **highly correlated features**, as it combines the strengths of both regularization methods.


Neural Networks are prone to overfitting particularly as the number of parameters increases. In this context, regularisation is any technique that produces better results on the test set.


The paper classifies regularisation techniques into four categories :
- Sparse vector regularisation
- Sparse Matrix Regularisation
- Matrix Regularisation
- Manifold Regularisation

# Sparse vector regularisation
### Sparse vector-based regularisation
Used in Compression Sensing, Feature Selection, Sparse PCA, Sparse Signals Separation

**$l_1$ norm** is as follows :$$ \|\mathbf{x}\|_1 = \sum_{j=1}^{N} |x_j|, $$
However, the $𝑙_1$ norm regularization cause a bias for large coefficients. In recent years, there has been a trend to study sparse regularizations with less bias.

**$l_p$ norm** is of significant interest

$$
\|\mathbf{x}\|_p = \left( \sum_{j=1}^{N} |x_j|^p \right)^{\frac{1}{p}}
$$
As $p$ goes to zero, this measure becomes a count of the non-zeroes in $\mathbf{x}$.

# The regularisation strategies in deep learning

**Dropout Regularisation**
	In every training step, a neuron has probability $p$ of being active. $p$ is the dropout rate of the model.
	Once trained the layer outputs are multiplied by $(1-p)$ (or divide the neurons during training) . This improves sparsity in neural network weights.
	**Dropconnect** drops weights or individual connections instead of neurons.
	In **DropMaps**, for a training batch, a feature is kept with a probability 
**Early Stopping**
	![[Pasted image 20250228223930.png|600]]
	Use the parameters from when the model has the least validation set error.

**Data Augmentation**
	![[Pasted image 20250228224301.png]]
	The model can overfit a single type of sample, so we generates new training samples by applying transformations to the existing data. 
	Rotation, Flipping, Cropping, Contrast, Brightness etc. are all methods to achieve this. This can also be done for audio data and text. 
	It is a particularly effective technique for image classification.
	Disadvantages include increased memory, training time and transformation costs. Color transformations may cause loss of essential information.
	Rotations are okay for cat vs dog but not for 6 vs 9. Label's consistency post transformation must be preserved.
	Feature space augmentation performs transformations, not in the input space, but the feature space.
	SMOTE (Synthetic Minority Oversampling Technique) works as follows :
	- Identify $k$ nearest neighbours of every data points in the minority class
	- Sample required number of points on the lines joining them

**Batch Normalisation**
	Batch Normalization normalizes activations across mini-batches to reduce **internal covariate shift**, improving model stability and regularization.
	For an activation $x$ in a batch: $$ \hat{x} = \frac{x - \mu_B}{\sigma_B}, \quad y = \gamma \hat{x} + \beta $$ where: 
	- $\mu_B$ is the **batch mean**. 
	- $\sigma_B$ is the **batch standard deviation**. 
	- $\gamma$ and $\beta$ are **learnable scaling and shifting parameters**.
	It prevents exploding or vanishing gradients since all data is normalised, hence preventing large weights in the network.