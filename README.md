# VehicleClassifierAdvanced
## What is this model?
### What is this model for?
This is a vehivle classifier.
This model can distinguish car, train, ship, airplane, and bicycle.

### This model's score
Validation score is "0.850"
Validation loss is "0.389"

#### Accuracy
![model accuracy](./ModelImages/Accuracy.png)

#### Loss
![model accuracy](./ModelImages/Loss.png)

### This model's summary
![model summary](./ModelImages/ModelSummary.png)

## How to use this model?

1: clone this repository.
```
git clone https://github.com/DOALA0155/VehicleClassifierAdvanced.git
```

2: move to this project's directory.
```
cd VehicelClassifierAdvanced
```

3: add image that the model predict.
Add image to the folder ("./VehicleClassifierAdvanced/PredictImages")

4: run python file
```
python3 predict_model.py [image name]
```
```
example: python3 predict_model.py sample.jpg
```
conditions
・This model can predict only jpg image.
・This model can predict image that is in "PredictImages" folder.
・Please enter the image name excluding directory name ("./PredictImages")
