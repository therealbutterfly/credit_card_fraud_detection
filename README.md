
# Credit Card Fraud Detection
Credit card fraud is the most significant type of transaction fraud in Canada , and experienced a 71% increase between 2008 and 2014 . Particularly, as online activity has increased due to the COVID-19 Crisis, new types of fraud are taking advantage of the fear and anxiety.   According to the Canadian Anti-Fraud Centre, Canadians have been defrauded out of at least $1.2 million in coronavirus-related scams.  

Moreover, fraudsters specifically target vulnerable populations, like seniors or low-income families, who may not have the resources to find recourse or reimbursement for their losses. In Canada, there are guarantees that protect credit card users against liability for credit card fraud, but may sometimes not be eligible to receive reimbursement as these guarantees are conditional on certain consumer behaviors.  So, credit card fraud can have large negative impact on people, and this trend is only rising every year. 

Financial service providers also face huge costs due to credit card fraud. In 2018, financial institutions reimbursed $862 million to their Canadian credit card customers, representing the losses these customers suffered as a result of criminal activities.  There is also the additional cost of resources for manually identifying and investigating fraudulent transactions.

The research question tackled in this project is how the impact of fraudulent credit card transactions be reduced to both consumers and to financial service providers. Machine learning tools can be used to reduce losses in different ways: identifying fraud clusters (a grocery store with a faulty reader, or a fraudulent website), or identifying customers who are more vulnerable to certain types of credit card fraud, or even by creating natural language processing tools that can reduce the cost of investigating suspicious activity. 

The methodology that I propose to tackle this problem in this project is to use Machine Learning tools to accurately identify fraudulent transactions, by spotting an anomaly in the pattern of credit usage.  This identification will allow financial service providers to limit greater losses by freezing credit cards before more fraudulent purchases can be made. Supervised classification algorithms like k-Nearest Neighbors, Classification Trees, and SVMs are examples of algorithms that can be applied to solve this problem. 


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
