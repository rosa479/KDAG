# KDAG Associates’ Selection Round 2

## Hola!
This is repo stores my submission for the second round of selection process for Kharagpur Data Analytics Group (KDAG) Associates.
The first round was a quiz round. The second round involved two coding tasks and a reading task. The coding tasks were Langton's Ants (Basic Python Proficiency) and Music Genre Analysis (NLP), overview of which can be found below. The Jupyter notebook for both tasks are well documented and the Music Genre Analysis report can be found [here](Task/Core%20Team%20-%20Round%202%20-%20Rohan%20Kumar%20Sah%20-%2024ME10134/Task%202/Report.pdf). The `Papers` folder is an Obsidian Vault containing my notes on the reading tasks. Given the fact that I got in, I think a did a decent job :D

## Core Team: Coding Task Overview

You are required to complete **both Task 1 and Task 2**.

- Submit your solution in `.ipynb` (Jupyter Notebook) format.
- Your notebook must run without errors using "Run All".
- Only use libraries explicitly mentioned in the task.
- Explain your thought process using **Markdown cells**.
- Final selection for interviews is based entirely on performance in these tasks.

---

## Task 1: Langton’s Ants

Langton’s Ant is a two-dimensional Turing machine simulation where ants move across a grid based on square colors and pheromones. Their simple movement rules lead to complex behavior over time.

### Movement Rules

1. **White Square**
   - Turn 90° clockwise
   - Flip the square color
   - Drop a pheromone (e.g., "A" or "B")
   - Move forward one unit

2. **Black Square**
   - Turn 90° counter-clockwise
   - Flip the square color
   - Drop a pheromone
   - Move forward one unit

### Pheromone Behavior

- **Self-pheromone**: 80% chance to move straight, 20% to follow standard turning rule
- **Cross-pheromone**: 20% chance to move straight, 80% to follow standard turning rule
- **Pheromone Replacement**: A new pheromone replaces any existing one
- **Pheromone Decay**: Influence fades over ~5 steps

### Technical Requirements

- Use **Python** only
- A simulation interface must be present (Pygame is allowed)
- Bonus points for using Object-Oriented Programming (OOP)

### Deliverables

- A `.py` file with runnable simulation code
- Code must be structured and readable

### Resources

- [Langton’s Ant - Wikipedia](https://en.wikipedia.org/wiki/Langton's_ant)  
- [Langton’s Ant Simulation - YouTube](https://youtu.be/-l_dDYriTPo?si=20co9FS0woBNQLvc)  
- [PyGame Documentation](https://www.pygame.org/docs/)

---

## Task 2: Music Genre Analysis

Analyze a dataset of songs tagged with three descriptive keywords and a genre label. The objective is to group songs based on their keyword similarities and extract insights.

**Dataset:**  
[Download Dataset](https://drive.google.com/file/d/168XF0_JH01azSPuOc8KvRV4j42Fyhw5N/view?usp=sharing)

### Rules

- Only use `numpy`, `pandas`, `matplotlib`, `seaborn`
- No use of `scikit-learn` or similar libraries for algorithm implementation

### Pipeline

#### 1. Keyword Vectorization

- Use both **BoW (Bag of Words)** and **TF-IDF**
- Compare both techniques and justify your choice
- Vectorize the keywords accordingly

Resources:
- [Bag of Words Introduction](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
- [TF-IDF Guide](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide)
- [Vector Embeddings - Computerphile](https://www.youtube.com/watch?v=gQddtTdmG_8)

#### 2. Dimensionality Reduction

- Implement **PCA (Principal Component Analysis)** manually using numpy
- Reduce vectors to 2D for each keyword type

Resources:
- [StatQuest: PCA Explanation](https://youtu.be/FgakZw6K1QQ?si=7QQvt4sl8cC9MJLI)

#### 3. Combine Embeddings

- Combine the reduced vectors into one embedding per song
- Suggested methods: averaging, concatenation, cross-product, etc.
- Justify your method

#### 4. Clustering

- Apply **K-Means** or other clustering method
- Justify your choice of `k` or the method itself
- Visualize clusters

Resources:
- [K-Means Clustering - StatQuest](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [Stanford CS229 Lecture on K-Means](https://www.youtube.com/watch?v=LmpkKwsyQj4&t=1083s)

#### 5. Final Analysis

- What is the genre distribution per cluster?
- Do clusters align with genre labels?
- Calculate **Silhouette Score**
- Predict genre for:
  - `[piano, calm, slow]`
  - `[guitar, emotional, distorted]`
  - `[synth, mellow, distorted]`

#### 6. Report

- Use a white-paper format
- Explain thought process, methodology, results
- Include plots and visualizations

#### 7. Bonus Section (Optional)

- Invent a creative vectorization technique
- Deeper analysis: by genre, cluster, or keyword
- Explore **extrinsic** and **intrinsic** clustering metrics

Example idea (not to be used directly):  
Create a 26D vector with frequency of each letter in keywords

### Final Submission Requirements

- A Jupyter Notebook (`.ipynb`), self-contained and error-free
- A report in PDF format

Additional Resources:
- [Clustering in Scikit-learn](https://scikit-learn.org/stable/modules/clustering)
- [K-Means++ Paper](https://youtu.be/NDAVDRFMh_0?si=LirfEz_l27sbsBf8)

---

## Reading Task

These papers are essential for your personal interview preparation. Choose at least one to prepare. Additional points will be awarded for preparing more than one.

- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://drive.google.com/file/d/1ql7X4IkcsjGcjgsnnhp5M3OzAZZmK-wt/view?usp=sharing)
- [A comprehensive survey on regularization strategies in machine learning](https://drive.google.com/file/d/1QBAhh9rYxSHSScTzGDWRg620XMqp1-0s/view?usp=sharing)
- [Statistical Modeling : The Two Cultures](https://drive.google.com/file/d/1SvWju8qw9K5sn1TKrc-oWjq_85uYBhB4/view?usp=sharing)

---

## Recommended Resources for Beginners

If you're not familiar with Python or the required libraries, refer to:

- [Scientific Computing with Python - freeCodeCamp](https://www.freecodecamp.org/learn/scientific-computing-with-python/)
- [Python Tutorial - W3Schools](https://www.w3schools.com/python/)
- [Numpy, Pandas, Matplotlib, Seaborn - YouTube](https://www.youtube.com/watch?v=r-uOLxNrNk8)

