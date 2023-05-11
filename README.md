# DNNLP_assignment_22_23_SN22082567

<p align="center">ELEC0141 Deep Learning for Natural Language Processing (DLNLP), Student Number: 22082567</p>

The objective of this project is to categorize English Twitter messages into six basic emotions: sadness, joy, love, anger, fear and surprise. Four models were used for classification. A used traditional machine learning model, Decision Tree. B, C, and D employed more complex artificial neural network (ANN) models built based on TensorFlow, including Multilayer Perceptron (MLP), Long Short-Term Memory (LSTM), and Bidirectional Long Short-Term Memory (BILSTM).

## Requirements
Python 3.9.16

  \- numpy==1.24.3   
  \- keras==2.9.0  
  \- matplotlib==3.7.1  
  \- pandas==2.0.1  
  \- scipy==1.10.1  
  \- scikit-learn==1.2.2  
  \- tensorflow==2.9.0  
  \- nltk==3.8.1  
  \- tensorflow==2.9.0  
  \- seaborn==0.12.2  
  \- tqdm==4.65.0
  
  \- cudnn==8.2  
  \- cudatoolkit==11.3  
  
More details at [requirements.txt](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/requirement.txt) and [environment.yml](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/environment.yml).

## Datasets
[Emotion Dataset for Emotion Recognition Tasks](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/Datasets) obtained from [Kaggle](https://www.kaggle.com/datasets/parulpandey/emotion-dataset) contains English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

[training.csv](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/Datasets/training.csv) contains 16,000 training tweets  
[validation.csv](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/Datasets/validation.csv) contains 2,000 validation tweets  
[test.csv](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/Datasets/test.csv) contains 2,000 test tweets  

## Folders and Files
- [A](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/A) is the code for Task A using decision tree model.  
- [B](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/B) is the code for Task B using MLP model.  
- [C](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/C) is the code for Task C using LSTM model.  
- [D](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/C) is the code for Task D using BILSTM model.  
- [Datasets](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/Datasets) is the datasets of this project.  
- [Jupyter Code](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/Jupyter%20Code) is the early development of this project using Jupyter testing.  
- [Results](https://github.com/Ys1ong/DLNLP_23_SN22082567/tree/main/Results) stores the results of this project.  
- [main.py](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/main.py) is to run the whole project.  
- [nlp.py](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/NLP.py) stores the function used for Task A,B,C and D.  

## Reports
Click here to see the full [report](https://github.com/Ys1ong/DLNLP_23_SN22082567/blob/main/Results/Deep_Learning_for_Natural_Language_Processing__ELEC0141__23_report.pdf).

## Results
<table>
  <tr>
    <td>Task</td>
    <td>Model</td>
    <td>Training Accuracy</td>
    <td>Validation Accuracy</td>
    <td>Test Accuracy</td>
    <td>Training Time</td>
  </tr>
  <tr>
    <td>A</td>
    <td>Decision Tree</td>
    <td>0.8930</td>
    <td>0.8597</td>
    <td>0.8420</td>
    <td>5.94s</td>
  </tr>
  <tr>
    <td>B</td>
    <td>MLP</td>
    <td>0.9844</td>
    <td>0.8705</td>
    <td>0.8760</td>
    <td>15.72s</td>
  </tr>
  <tr>
    <td>C</td>
    <td>LSTM</td>
    <td>0.9796</td>
    <td>0.8885</td>
    <td>0.8870</td>
    <td>117.54s</td>
  </tr>
  <tr>
    <td>D</td>
    <td>BILSTM</td>
    <td>0.9891</td>
    <td>0.8865</td>
    <td>0.9125</td>
    <td>88.38s</td>
  </tr>
</table>
