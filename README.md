
# Credit Card Fraud Detection
Credit card fraud is the most significant type of transaction fraud in Canada , and experienced a 71% increase between 2008 and 2014. The main objective of this project is to use machine learning tools to detect fraudulent credit card transactions. By accurately identifying fraudulent credit card transactions, financial institutions can limit losses to consumers by freezing or suspending the credit cards.

The research question tackled in this project is “Is a given credit card transaction fraudulent?”, which is a question that falls under the overarching theme of Anomaly Detection Problems. To answer this question, the Credit Card Fraud Detection Dataset, available on Kaggle.com , will be used to build a machine learning model. The data contains around 280,000 European credit card transactions, labelled as either fraudulent or non-fraudulent. 

Supervised classification algorithms like k-Nearest Neighbors, Classification Trees, and SVMs are popular models used to solve similar problems. Additionally, techniques to balance the data and split into the test/training sets will be used to avoid overfitting / overtraining the model. The main tool used to employ these techniques will be Python, leveraging resources such as the Numpy, Pandas, Scipy, and Scikit-learn libraries.


## Data source

The open source data used in this project is the [MLB-ULB Credit Card Transaction Data](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Guide

The different aspects of this project have been broken down into different .py files:

1. Explore data.py 
2. Feature Selection.py
3. Data Balancing.py
4. Models.py

Please note that since the dataset used is pretty large, running code can take some time, especially the wrapper-type Feature Selection models. 

## Project Next Steps

1. Identify appropriate performance measures to evaluate models
2. Tweak parameters and run models, with different combinations of feature selection and data balancing tools
3. Identify "winning" model combination

## Acknowledgements

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019
