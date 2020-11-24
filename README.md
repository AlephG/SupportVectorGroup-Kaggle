# SVG-Kaggle
Kaggle Competition for MAIS202 @McGill

## MultiDigit MNIST Classifier

Detailed proprocessing and model implementation and outputs can be found in the Jupyter Notebook included in this repo as well as in the included write-up. Class predictions can be replicated by running the included Jupyter notebook.

### Data Preprocessing

Pixels below a threshold of 235 were set to 0, this removed most noise from the images and only outlines of the hand drawn digits remained. We define a function `clean` in order to apply this to all pixels in the dataset.

```
def clean(img):
  _, thresh = cv2.threshold(img.astype(np.uint8), 235, 255, cv2.THRESH_BINARY)
  thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB) #pseudo RGB image
  return thresh
```

Cleaned images are then augemented using the `keras` implementation of a `ImageDataGenerator` in order to diversify the training set.

### Xception Model

The Xception CNN architecture used in this classification problem is the `keras` provided implementation with an early stopping callback, a learning rate scheduler and stochastic gradient descent via Adam and trained for 40 epochs.

### Results

Classification on training data resulted in 95% accuracy and classification on unseen the test set resulted in 96% accuracy.
