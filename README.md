# Big Data experimental Thesis - University of Catania

## Thesis Title
**A Graph Convolutional Network-based model for the stratification of patients with non-communicable diseases**

## Project Overview
This repository contains the experimental thesis project for the Master of Data Science at the University of Catania. The focus of this project is on applying **Graph Convolutional Network** to tackle the challenge of **the stratification of patients with non-communicable diseases**. 
The goal of the model aims to classify breast cancer patients according to their prognosis, distinguishing between short- and long-term survival.

![alt text](https://github.com/francescogra/MasterThesisExam/blob/main/slide3.jpg "GCN")

Breast cancer represents the most common form of cancer among women, with an increasing need for reliable algorithms to predict its prognosis. 
In this context, the present study proposes a model based on a convolutional graph network (GCN) incorporating clinical data from the METABRIC database.

Using graphs to extract structural information, the model aims to classify breast cancer patients according to their prognosis, distinguishing between short- and long-term survival.
The model exploits advanced graph representation and convolution techniques to construct a normalised adjacency matrix representing the network of gene interactions. For each patient and gene, a normal distribution is constructed based on the detected gene expression values, sampling repeatedly to obtain a three-dimensional matrix. Subsequently, a GCN-based autoencoder reduces the size of these matrices, creating a new network of patient interactions.
This network is crucial for the next classification step, where patients are categorised according to their short- or long-term survival. The model thus constructed is then evaluated through a series of performance metrics to ensure its accuracy and reliability.

# Datasets

![alt text](https://github.com/francescogra/MasterThesisExam/blob/main/slide1.png "Dataset")

The METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset, a widely recognised body of genomic and clinical data in the field of breast cancer research, was used for this project. It provides a comprehensive molecular characterisation of over 2000 tumours.
Specifically, the analysis focused on the METABRIC gene expression data, which provide a quantitative representation of the transcriptional activity of thousands of genes in each tumour sample.
At a biological level, genes are segments of DNA that contain instructions for building proteins, while gene expression determines which proteins are produced in a cell, and thus which functions the cell can perform.
Since in cancer, gene expression patterns are often altered, comparing healthy patients with non-healthy patients can help identify potential therapeutic targets.
The approach that was adopted laid its foundations on a detailed and thorough understanding of the datasets, an essential prerequisite for developing the code in a manner channelled to the intended goal. The first steps were in fact to examine in detail the structure and composition of the datasets, their nature and typology.
This preliminary phase proved crucial in order to acquire a solid knowledge of the available information and its interconnections.


# The following is a preliminary description of all the steps in the algorithm:

1. The gene data are pre-processed to correct the format and normalise the values. After removing unnecessary columns and replacing non-standard characters, a data matrix is created that includes the patients' gene expressions. These data are then used to generate random samples based on the normal distribution, representing the patients' gene expressions.

2. Next, the K-Means algorithm is applied to perform clustering by transforming and normalising the data to facilitate analysis. Then clustering is performed on the pre-processed data, identifying the optimal number of clusters through the elbow method and Silhouette Score. The obtained clusters are then used to create a patient network, where the nodes represent the patients and the arcs the similarity relationships between them.

3. This network is represented using the NetworkX library, and the nodes are coloured according to the survival of the patients.

4. Next, the patient network, gene expression data and survival labels are loaded. The network adjacency matrix is normalised, and gene features are transformed by Z-score normalisation. The dataset is divided into training, validation and test sets.

5. The GCN model is defined with several convolutional layers to extract meaningful features from the network data. The network is trained to classify patient survival, using a cross-entropy loss function and the Adam optimiser. The training takes place at different epochs, and for each epoch, the loss and accuracy on both the training and validation sets are calculated. The model is evaluated periodically for performance improvement.

6. Finally, after training, the model is tested using the test set to assess its ability to generalise to unseen data.


![alt text](https://github.com/francescogra/MasterThesisExam/blob/main/slide2.png "patients Network")


# Results and conclusion


Initially, analysis of the METABRIC dataset revealed a strong imbalance between survival classes, with 183 samples for OS>=2 and only 30 for OS<2. This imbalance required the implementation of balancing techniques, in particular the use of SMOTE, which resulted in an increase in minority class samples from 30 to 183, creating a more balanced dataset of 366 total samples.

Another critical aspect was the optimisation of the patient network structure. By applying the Elbow and Silhouette Score methods, the optimal number of clusters for the K-means algorithm was reduced from 50 to 12, significantly improving the quality of patient segmentation and increasing the number of connections in the network from 2200 to 3000 nodes.

Experimentation with different optimisers and activation functions produced remarkable results. In particular, the Adagrad optimiser showed the best performance, with a test accuracy of 94.41% and a loss of 0.191. Among the activation functions, torch.log_softmax proved to be the most effective, with a test accuracy of 92.47% and a loss of 0.216.

The introduction of the adjacency matrix normalisation in the GCN forward function was a turning point that significantly improved the performance of the model. This change increased the training accuracy from 90.03% to 94.58% and reduced the training loss from 0.266 to 0.149.

In the final optimised configuration, the model achieved excellent results on the test set with a loss of 0.0965, an accuracy of 96.36%, an F1 score of 0.9636, an accuracy of 96.43% and a recall of 96.36%. These results demonstrate the robustness and effectiveness of the proposed model in stratifying breast cancer patients.

Comparison with other graph-based models, such as GraphSAGE and GAT, highlighted the potential of GCN. In particular, GraphSAGE showed comparable performance, suggesting that it could be a viable alternative in certain contexts.
