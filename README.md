# filoVision: a platform that uses deep learning and tip markers to automate filopodia quantitation

<img src="https://user-images.githubusercontent.com/67563125/228630157-8abc693a-b55b-4c0b-864a-ec48a7ea4582.jpg" width="50%" height="50%">

<img src='https://user-images.githubusercontent.com/67563125/228630382-f2ffdbf9-6220-4900-ae03-11e6e9f64756.jpg' width=50% height=50%>

## Description
filoVision contains two Google Colab notebooks, ["filoTips"](https://colab.research.google.com/drive/1mL7U63-lltjMoTKgpcUbhK-iV0GoYz3L) and ["filoSkeleton"](https://colab.research.google.com/drive/1-61DTdWYXMIecJqjE5nWMFue2JJBbuBj), used to automate the tedious act of quantifying filopodia. filoTips was developed to analyze the filopodia of cells expressing a fluorescent tip marker without requiring a labeled cytoskeleton. filoSkeleton, on the other hand, combines tip markers and actin labeling for when either labeling method alone is unsuccessful.

## Who should use this?
Anyone who spends a considerable amount of time quantifying filopodia and commonly expresses fluorescent tip markers in their experiments. They should also be willing to spend initial time training a deep learning model using ZeroCostDL4Mic.

## Getting Started
#### 1. Trial run a filoVision notebook using the evaluation data and default models (not required - click 'Code' and 'Download Zip' to download images)
#### 2. Train custom filoVision models on the user's data with the [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic) framework
#### 3. Plug a custom model into one of the filoVision notebooks
#### 4. Run a filoVision notebook on future data for automated filopodia quantitation

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

#### Margaret A. Titus
titus004@umn.edu

## Acknowledgments
We would like to express our gratitude to Karl Petersen for creating the SEVEN imageJ macro, which served as inspiration for the filoVision platform, specifically filoTips. We are grateful to Ashley Arthur and Samuel Gonzalez for introducing us to Ilastik and the ZeroCostDL4Mic platform, respectively, which have greatly facilitated the training of custom filoVision models. We are also grateful to the majority of former members of the Titus lab whose raw imaging data contributed to the training dataset for the filoTips model. Additionally, we would like to thank Amy Crawford, Emma Hall, and Taylor Sulerud for testing and providing feedback on the filoVision platform, which helped us to troubleshoot and improve the platform. Finally, we would like to acknowledge the developers of ZeroCostDL4Mic, Google Colab, TensorFlow, Keras, and OpenCV, whose tools have made the development of filoVision possible.


