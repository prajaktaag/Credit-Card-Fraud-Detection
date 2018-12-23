# Credit-Card-Fraud-Detection
Machine learning models to automatically predict credit card frauds

### Dataset ###
The dataset for this project was obtained from kaggle website.  
[https://www.kaggle.com/mlg-ulb/creditcardfraud]

### Requirements ###
* Python 3.6 
* Pandas and Numpy
* Keras 2.1.6
* TensorFlow 1.8
* Sklearn Library 0.19.1
* Python's Matplotlib (2.2.2) for Visualization

### Exploratory Data Analysis ###

       1. Checking the target class. (0 - No Fraud  1 - Fraud).

<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40387207-fcc221ee-5dd9-11e8-9e1a-33bf9c2d22bf.png" width="400" height = "380" /> </p>

       2. Checking Time of Transaction vs Amount of Transaction for each class ((0 - No Fraud  1 - Fraud)

<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40451983-65a935ec-5eae-11e8-9125-9849e37aa1fc.png" width="350" height = "350" /> </p>

       3.  Amount per Transaction for each class (0 - No Fraud  1 - Fraud)
       
<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40451984-65c47c76-5eae-11e8-914a-0f79ef3e86f1.png"  width="350" height = "350" />
 </p>

       4. Correlation Matrix between different features of the dataset

<p align = "center" > 
<img src = "https://user-images.githubusercontent.com/38739938/40387208-fcd07e7e-5dd9-11e8-8eab-4af46c01425a.png" width="300" height = "300" align = "center"/> </p>

       3. Describing the dataset 
<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40387209-fcdfc8ca-5dd9-11e8-88e2-ad830e554857.png" width= "250" height = "250" align = "center"/>
</p>
       

## Data Preprocessing ##
1.  Feature Elimination 
2.  Data Normalization 
3.  Balancing the skewed dataset using  
       3.1  Undersampling technique     
       3.2 Oversampling using SMOTE technique    
  
## Machine Learning Models ##
Used Sklearn, TensorFlow and Keras libraries to built the following models in Spyder IDE. 

1. Autoencoder Artificial Neural Networks  
2. Random Forest
3. Logistic Rregression

## Performace Evaluation ##
1. 5 fold cross Validation
2. Confusion Matrix
3. Precision - Recall Curves
4. Cohen's Kappa Statistic
5. AUC - ROC curves
  
##  Results ##


### 1.  Autoencoders -  Artificial Neural Network ###

       1.1 Model Loss for Autoencoders  -  Plot of Loss vs Epoch of training and testing data
 _Epoch = 88 , Batch_size = 32_
 
 <p align = "center">
 <img src="https://user-images.githubusercontent.com/38739938/40451986-65e9eb50-5eae-11e8-8f6c-5fe988a5bec1.png" width = "380" height = "320"/> </p>
 
       1.2 Error Distribution of Autoencoders for each class (0 - No Fraud  1 - Fraud).
       
<p align = "center">    
<img src="https://user-images.githubusercontent.com/38739938/40451993-66668818-5eae-11e8-867c-42a05c0ce51b.png" width = "300" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40451987-65fa0b02-5eae-11e8-9535-a7dec497f14f.png" width = "250" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40451988-660ad5e0-5eae-11e8-814b-57fe56d6a75e.png" width = "250" height = "250"/>  </p>
                <p align = "right"> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  reconstruction error with no fraud  &nbsp &nbsp &nbsp &nbsp reconstruction error with fraud </p>

      1.3 AUC - ROC  and Confusion Matrix for Autoencoders Neural Nets
<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40451989-6620abcc-5eae-11e8-8d51-ef9d17423e66.png" width = "300" height = "260"/>  <img src="https://user-images.githubusercontent.com/38739938/40451994-6675c166-5eae-11e8-9631-b9565f9a3c7c.png" width = "300" height = "260"/> 

      1.3  Precision- Recall Curves for Autoencoders Neural Nets
<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40451990-663605e4-5eae-11e8-9942-f381c1294848.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40451991-6645c43e-5eae-11e8-8cc4-c6974a629558.png" width = "280" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40451992-6652fa00-5eae-11e8-994c-5261bb63a5b0.png" width = "280" height = "250"/> </p>


### 2.  Random Forest ###
         2.1  AUC - ROC for Random Forest
<p align = "center">
<img src="https://user-images.githubusercontent.com/38739938/40388626-0b5b2990-5dde-11e8-8a08-28dcf1465989.png" width = "280" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40388627-0b6a6860-5dde-11e8-8597-d524e2dbe461.png" width = "280" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40388628-0b7926c0-5dde-11e8-9fe3-33581093d1c1.png" width = "280" height = "250"/> </p>

       2.2  Precision- Recall Curves for Random Forest
<img src="https://user-images.githubusercontent.com/38739938/40388632-0bc7d6c6-5dde-11e8-91a5-d233fbee0d48.png" width = "280" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40388633-0bda58dc-5dde-11e8-8c8a-448f14fb61e0.png" width = "280" height = "250"/> <img src="https://user-images.githubusercontent.com/38739938/40388634-0bebc284-5dde-11e8-973f-ab469b62c593.png" width = "280" height = "250"/>

       2.3 Confusion Matrix for Random Forest
<img src="https://user-images.githubusercontent.com/38739938/40388629-0b8b246a-5dde-11e8-9631-23da1df36b86.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388630-0ba49a26-5dde-11e8-9b0b-82bfac55b50c.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388631-0bb4ed54-5dde-11e8-90e4-cb3f9c7a3f5b.png" width = "280" height = "250"/>


### 3.  Logistic Rregression ###

         3.1  AUC - ROC for Logistic Rregression
<img src="https://user-images.githubusercontent.com/38739938/40388655-15a1292c-5dde-11e8-818a-455524c90de8.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388656-15b25e5e-5dde-11e8-8cbc-9a4ca9d80037.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388657-15ccc6c2-5dde-11e8-9a21-b76c291041a5.png" width = "280" height = "250"/>

         
         3.2  Precision- Recall Curves for Logistic Rregression
<img src="https://user-images.githubusercontent.com/38739938/40388652-15679360-5dde-11e8-9613-7ba4b576aa20.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388653-15792bac-5dde-11e8-9f91-518c1c88d79f.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388654-158bcc44-5dde-11e8-8ca8-9c6ba8eb5a5d.png" width = "280" height = "250"/>

           3.3  Confusion Matrix for Logistic Rregression
 <img src="https://user-images.githubusercontent.com/38739938/40388649-1533b626-5dde-11e8-85c0-995423b29624.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388650-1545fef8-5dde-11e8-9feb-d55ceb684dea.png" width = "280" height = "250"/><img src="https://user-images.githubusercontent.com/38739938/40388651-15582394-5dde-11e8-9cc7-2877995b4c93.png" width = "280" height = "250"/>




 
