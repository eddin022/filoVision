# filoVision: a platform that uses deep learning and tip markers to automate filopodia analysis
  
## Description
filoVision contains two Google Colab notebooks, ["filoTips"](https://colab.research.google.com/drive/1mL7U63-lltjMoTKgpcUbhK-iV0GoYz3L) and ["filoSkeleton"](https://colab.research.google.com/drive/1-61DTdWYXMIecJqjE5nWMFue2JJBbuBj), used to automate the tedious act of quantifying filopodia. filoTips was developed to analyze the filopodia of cells expressing a fluorescent tip marker alone without requiring a labeled cytoskeleton. filoSkeleton, on the other hand, combines a tip marker and an actin label for when more precise stalk lengths of long, curved filopodia are needed.

<img src='https://user-images.githubusercontent.com/67563125/228642549-6a17d475-ebde-4338-a0d7-ed9ca3435fb1.jpg' />

## Who should use this?
Anyone who spends a considerable amount of time quantifying filopodia and commonly expresses fluorescent tip markers in their experiments. Users are free to use our default models, but we recommend they tune our models to their own data with transfer learning via the ZeroCostDL4Mic platform.

## Getting Started
### Recommended trial run using the example data provided here
#### 1. Click 'Code' and 'Download Zip' to download the repository which contains the example data
#### 2. Click the filoTips or filoSkeleton Google Colab link above
#### 3. Drag the example data into the "Files" section of the notebook
#### 4. Provide the requested inputs
#### 5. Click "Runtime" and "Run All"
#### 6. Your filoVision analysis will be downloaded locally to your system once complete

### Analyzing user data
#### 1. Follow above, but with user data instead of example data
#### 2. If using a tuned model, select "Custom Model" and provide a Google Drive link to the model
#### 3. If the user wants to tune our models to their data, they should use the [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) platform

## Acquiring filoVision Models and Train/Test Data
#### Automatically
If a default filoVision model is selected during filoVision analysis, the model is automatically downloaded. There is no need for the user to take additional steps.
#### Bioimage.io or Zenodo
The default filoVision models and train/test data will also be available on Bioimage.IO and Zenodo. Currently they are under review, but links will be posted here as they become available on these platforms.

### Dependencies
#### ZeroCostDL4Mic
Training a deep learning model on representative data is crucial for proper accuracy with just about any deep learning tool. Training these models can be challenging for those with minimal experience with libraries like TensorFlow and Keras. Thus, we exploited the impressive "ZeroCostDL4Mic" framework to empower anyone with the ability to train their own model and plug it into filoVision.

#### Hardware and software
Because the notebooks were developed in a Google Colab environment, the hardware and software dependencies for filoVision are minimal. Since filoVision runs in the cloud, most modern computers should be able to run it as long as they have access to the internet.

## How to cite this work

## Help
The most common source of error for new users is properly naming their files. If the user is running into errors, the first recommendation is to check that the files being run are named according to the instructions in the appropriate notebook.

## Authors
#### Casey Eddington
eddin022@umn.edu

#### Jessica Schwartz
jaschwar@umn.edu

#### Margaret A. Titus
titus004@umn.edu

## Acknowledgments
We would like to express our gratitude to Karl Petersen for creating the SEVEN imageJ macro, which served as inspiration for the filoVision platform, specifically filoTips. We are grateful to Ashley Arthur and Samuel Gonzalez for introducing us to Ilastik and the ZeroCostDL4Mic platform, respectively, which have greatly facilitated the training of custom filoVision models. We are also grateful to the majority of former members of the Titus lab whose raw imaging data contributed to the training dataset for the filoTips model. Additionally, we would like to thank Amy Crawford, Emma Hall, and Taylor Sulerud for testing and providing feedback on the filoVision platform, which helped us to troubleshoot and improve the platform. Finally, we would like to acknowledge the developers of ZeroCostDL4Mic, Google Colab, TensorFlow, Keras, and OpenCV, whose tools have made the development of filoVision possible.


