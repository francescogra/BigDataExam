import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')  # Utilizza il backend TkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import networkx as nx  # Importa networkx
from imblearn.over_sampling import SMOTE

# Read the CSV file with the specified separator
df = pd.read_csv("D:\\uni\\magistrale\\materie\\Tesi\\GeneExpressionMetabric.csv", sep=';')

# Estrai gli ID dei pazienti
patient_ids = df.iloc[:, 1]

# Controllo il bilanciamento delle classi prima di SMOTE
class_labels = df.iloc[:, -1]
class_counts = class_labels.value_counts()
print("Occorrenze delle classi prima di SMOTE:")
print(class_counts)

data_genes = df.iloc[:, 1:-3]
data_genes_smote = df.iloc[:, 2:-3]
data_survival = df.iloc[:, -1]  # L'ultima colonna contiene le informazioni di sopravvivenza

data_genes_smote = data_genes_smote.applymap(lambda x: float(str(x).replace(',', '.')))

# Convert DataFrame to a NumPy array
data_smote = data_genes_smote.to_numpy()
data = data_genes.to_numpy()

# Converti le etichette di classe in numeri
class_mapping = {'OS>=2': 1, 'OS<2': 0}
data_survival_numeric = data_survival.map(class_mapping)

# Applica SMOTE per bilanciare il dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(data_smote, data_survival_numeric)

# Ripristina gli ID dei pazienti per i dati sintetizzati
# Creiamo un DataFrame temporaneo con i dati sintetizzati
resampled_df = pd.DataFrame(X_resampled, columns=data_genes.columns[1:])

# Ricrea gli ID pazienti per i nuovi campioni sintetici
n_new_samples = X_resampled.shape[0] - data.shape[0]
new_patient_ids = [f"MB-{7000 + i}" for i in range(n_new_samples)]

# Aggiungi i nuovi ID pazienti alla lista degli ID esistenti
all_patient_ids = pd.concat([patient_ids, pd.Series(new_patient_ids)], ignore_index=True)

# Aggiungi la colonna degli ID pazienti al DataFrame sintetizzato
resampled_df.insert(0, 'Patient_ID', all_patient_ids)

# Aggiungi la colonna delle etichette di sopravvivenza al DataFrame sintetizzato
resampled_df['Survival'] = pd.Series(y_resampled).map({1: 'OS>=2', 0: 'OS<2'})


# Ora X_resampled e y_resampled contengono il dataset bilanciato
print("Numero di campioni prima dell'applicazione di SMOTE:", data.shape[0])
print("Numero di campioni dopo l'applicazione di SMote:", X_resampled.shape[0])

# Verifica il bilanciamento delle classi dopo SMOTE
class_counts_after_smote = resampled_df['Survival'].value_counts()
print("Occorrenze delle classi dopo SMOTE:")
print(class_counts_after_smote)


data_genes_new = resampled_df.iloc[:, 1:200]
data_survival_new = resampled_df.iloc[:, -1]  # L'ultima colonna contiene le informazioni di sopravvivenza

# Convert DataFrame to a NumPy array
data_new = data_genes_new.to_numpy()

# Ensure the entire array is of type string for string operations
data_new = data_new.astype(str)

# Replace all occurrences of the comma with a period in the entire array
data_new = np.char.replace(data_new, ',', '.')

# Convert the remaining columns to float after replacing commas with periods
data_new[:, 1:] = data_new[:, 1:].astype(np.float64)


# Define a function to replace "MB-" with an empty string
def replace_func(value):
    return value.replace("MB-", "")

# Create a vectorized version of the function
vfunc = np.vectorize(replace_func)

# Apply the vectorized function to the first column of the matrix
data_new[:, 0] = vfunc(data_new[:, 0])

# Save the patient id column
patients_id = data_new[:, 0]

# Put them in a row
patients_id_inc = patients_id.reshape(-1, 1)

n_samples = 500

# Create an empty list to store matrix
patient_gene_arrays = []

print("Cycle on patients")

# Cycle on patients
for i in range(data_new.shape[0]):
    patient_matrix = []
    # Cycle on genes
    for j in range(1, data_new.shape[1]):  # Start from 1 to skip the patient ID column
        # Calculate gene standard deviation on all patients
        sd_gene = np.std(data_new[:, j].astype(np.float64))  # Ensure the data is float
        # Generates a normal distribution of 1000 elements for the current patient-gene pair
        samples = np.random.normal(float(data_new[i, j]), sd_gene, n_samples)
        patient_matrix.append(samples)
    patient_gene_arrays.append(patient_matrix)

# Convert the matrix list to a numpy three-dimensional array
patient_gene_arrays = np.array(patient_gene_arrays)

# Delete the first row of each matrix within data_new, that is, delete the id patients from the calculation
patient_gene_arrays_no_id = patient_gene_arrays[:, 1:, :]

# Apply mean to collapse the results in a gene
mean_matrix = []
for p in range(len(patient_gene_arrays_no_id)):
    mean_matrix.append(patient_gene_arrays_no_id[p].mean(axis=1))
    mean_matrix[p] = mean_matrix[p].reshape(-1, 1)

# Transform the list in numpy.ndarray
mean_matrix = np.stack(mean_matrix)

# This will create a new list transposed_arrays that contains the transposed versions of the arrays in the original list
transposed_arrays = [array for array in mean_matrix]

# Create a single matrix from numpy.ndarray of numpy.ndarrays
result = np.concatenate(transposed_arrays, axis=1)

# Transpose
result_T = result.transpose()

# Combine id patients with values
data_updated = np.concatenate((patients_id.reshape(-1, 1), result_T), axis=1)

# Flatten the matrix to 2d
data_2d = data_updated.reshape(data_updated.shape[0], -1)

# Remove patients (first column)
data_2d = data_2d[:, 1:]

# Convert to float to avoid dtype issues
data_2d = data_2d.astype(np.float64)

# Prova diversi valori di n_clusters
inertia = []
silhouette_scores = []
cluster_range = range(1, 51)  # Es. da 1 a 50 cluster
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(data_2d)
    inertia.append(kmeans.inertia_)
    if k > 1:  # Silhouette Score ha senso solo per k>1
        score = silhouette_score(data_2d, kmeans.labels_)
        silhouette_scores.append(score)

# Traccia il grafico del metodo del gomito
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Traccia il grafico del Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(cluster_range[1:], silhouette_scores, marker='o')  # Skipping the first element as Silhouette Score starts from k=2
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Scegli il numero ottimale di cluster basato sui grafici
optimal_clusters = 12  # Cambia questo valore in base ai grafici

# Create an instance of KMeans con il numero ottimale di cluster
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, n_init=10)

clusters_per_patient = []

# Fit the model to the data matrix
kmeans.fit(data_2d)

# Predict clusters for each point
clusters = kmeans.predict(data_2d)

clusters_per_patient.append(clusters)

# Convert id_patients, which is the row of id patients, to a 2-dimensional shape
patients_id_cluster = patients_id_inc.reshape(1, -1)

patients_id_cluster = np.append(patients_id_cluster, clusters_per_patient, axis=0)

print("Create a dictionary")

# Create a dictionary to map each value (cluster) to a list of patient IDs
value_to_ids = {}
for id, value in zip(patients_id_cluster[0], patients_id_cluster[1]):
    if value not in value_to_ids:
        value_to_ids[value] = []
    value_to_ids[value].append(id)

# Create a dictionary to map each id pair to a weight
id_pair_to_weight = {}
for ids in value_to_ids.values():
    if len(ids) > 1:
        for i, id1 in enumerate(ids):
            for id2 in ids[i + 1:]:
                pair = tuple(sorted([id1, id2]))
                weight = sum(patients_id_cluster[1][patients_id_cluster[0] == id1] == patients_id_cluster[1][patients_id_cluster[0] == id2])
                if pair in id_pair_to_weight:
                    id_pair_to_weight[pair] += weight
                else:
                    id_pair_to_weight[pair] = weight

# Create the network with weights as a two-dimensional array of tuples
network = np.array(list(id_pair_to_weight.keys()))

# Save the network to a file
np.savetxt("data_uni/network.txt", network, fmt='%s', delimiter="   ")

# Crea un grafo NetworkX dalla rete caricata
G = nx.Graph()
for edge in network:
    G.add_edge(edge[0], edge[1])

# Assicurati che ogni paziente sia aggiunto come nodo al grafo
for patient_id in patients_id:
    if patient_id not in G:
        G.add_node(patient_id)

# Colora i nodi in base alla sopravvivenza
node_colors = ['green' if survival.startswith('OS>=2') else 'red' for survival in data_survival_new]

# Verifica del numero di nodi e colori prima di disegnare il grafo
print(f"Numero di nodi nel grafo: {G.number_of_nodes()}")
print(f"Numero di colori: {len(node_colors)}")

# Disegna il grafo con i nodi colorati in base alla sopravvivenza
plt.figure(figsize=(20, 20))  # Aumenta la dimensione del grafico
pos = nx.spring_layout(G, k=1)  # Regola il parametro 'k' per aumentare la distanza tra i nodi
nx.draw(G, pos, with_labels=True, node_size=40, node_color=node_colors, font_size=8, font_weight='bold', edge_color='gray')  # Riduci la dimensione dei nodi e del testo
plt.title('Patient Network Graph')
plt.show()
