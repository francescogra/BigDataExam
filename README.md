# Master's thesis

# A Graph Convolutional Network-based model for the stratification of patients with non-communicable diseases

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
