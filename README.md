# decision_tree

## Couple of thoughts about this as i go along. First of all the model part of this problem is the most technical but its also the most solved so I will be working on it a bit towards the end. I think that building the framework around a project is important here.
### The model will be built from two classes Node and Model. The Node is where we can store all node data while the model will perform the recurrsion and allow for the application of the cost function Gini or Entropy.
### Notes about the data: I do recognize that this is a toy dataset so not exhaustive but i would like to point out some notes here. The data itself does need new features as the features of color and number of legs are clearly weak when it comes to the animal kingdom on the whole but we know that. Next I would point out that the categorical variables here (which number of legs technically is as well since the varable has a finite number of answers even if we include insects). 
### In an atempt to do a better job with our features given i would recommend (given time i might try it) that color at least is embedded using glove or other embedding models. One hot encoding or indexing here does not properly represent the distance between the colors and in future i could see this as an space to have access to (animals change color depending on diet, habitat, danger level, and where they are on the food chain hence it could be a good feature in the future)

### Also here i will not bother with a train test split since there is 4 data points.

