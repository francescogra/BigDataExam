import os
import shutil
import re

# Definire i percorsi delle cartelle
source_folder = "results"
destination_folder = "br"

# Funzione per copiare i file che soddisfano la condizione
def copia_file(source_folder, destination_folder):
    filenames = sorted(os.listdir(source_folder))
    for filename in filenames:
        if "loss_train" in filename:
            # Estrarre il valore di loss_train
            try:
                pattern = r"_loss_train=_([0-9.]+)_acc_train=_"
                match = re.search(pattern, filename)

                if match:
                    loss_train_value = float(match.group(1))
                    print("Valore di loss_train:", loss_train_value)

                    # Controllare se il valore di loss_train è inferiore a 0.15
                    if loss_train_value <= 0.15:
                        # Copiare il file nella cartella di destinazione
                        shutil.copy(os.path.join(source_folder, filename), destination_folder)
                        print(f"Copied: {filename}")
                else:
                    print("Nessuna corrispondenza trovata.")
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {e}")


# Eseguire la funzione
copia_file(source_folder, destination_folder)