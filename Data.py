from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pickle as pk
import os

from Proccessing import tj_class, string_to_dict, Graph


class Data:
    def __init__(self, data_folder, dataset_folder, normalisation_method):
        self.normalisation_method = normalisation_method
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.data = []

    def _load_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return pk.load(f)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

    def prepare_data(self):
        print("Getting Data ...")

        # Récupérer tous les fichiers correspondants dans le dossier
        files = sorted([f for f in os.listdir(self.data_folder) if f.startswith("data_") and f.endswith(".pkl")])
        i = 1
        for file in files:
            print(file, end=" ")
            filepath = os.path.join(self.data_folder, file)
            batch = self._load_file(filepath)
            if i == 16 :
                self.save_data()
                i = 1

            i += 1
            if batch:
                for b in batch:
                    row = {
                        "Jmax": b.get("Jmax"),
                        "t10": b.get("t10"),
                        "Deff": b.get("Deff"),
                        "C0eff": b.get("C0eff"),
                        "tj0r": string_to_dict(b.get("fractions")).get("tj0r"),
                        "tj1r": string_to_dict(b.get("fractions")).get("tj1r"),
                        "tj2r": string_to_dict(b.get("fractions")).get("tj2r"),
                        "tj3r": string_to_dict(b.get("fractions")).get("tj3r"),
                        "fGB": string_to_dict(b.get("fractions")).get("fGB"),
                        "tj": tj_class(b.get('TJ')),
                        "graph": Graph(b.get('adj_m'), b.get('Randoms'))
                    }
                    self.data.append(row)

        self.save_data()

    def normalise(self, method="min_max"):
        #print(f"Normalising using {method.replace('_', ' ').title()}...")

        features = ["Jmax", "t10", "Deff", "C0eff", "tj0r", "tj1r", "tj2r", "tj3r", "fGB"]
        values = np.array([[row[feature] for feature in features] for row in self.data])

        if method == "standard_score":
            scaler = StandardScaler()
        elif method == "min_max":
            scaler = MinMaxScaler()

        normalized_values = scaler.fit_transform(values)

        for i, row in enumerate(self.data):
            for j, feature in enumerate(features):
                row[feature] = normalized_values[i, j]

    def save_data(self):
        os.makedirs(self.dataset_folder, exist_ok=True)
        existing_files = [f for f in os.listdir(self.dataset_folder) if f.startswith("data_") and f.endswith(".pkl")]
        existing_indices = []

        for file in existing_files:
            try:
                index = int(file.split("_")[1].split(".")[0])
                existing_indices.append(index)
            except ValueError:
                continue

        if existing_indices:
            next_index = max(existing_indices) + 1
        else:
            next_index = 1

        print("\nSaving batch ", next_index)
        new_file_path = os.path.join(self.dataset_folder, f"data_{next_index}.pkl")
        os.makedirs(self.data_folder, exist_ok=True)
        with open(new_file_path, 'wb') as f:
            pk.dump(self.data, f)

        self.data = []

    def get_dataset(self, index) :
        file = os.path.join(self.dataset_folder, "data_" + str(index) + ".pkl")
        self.data = self._load_file(file)
        self.normalise(self.normalisation_method)

        features1 = ["Jmax", "t10", "Deff", "C0eff"]
        features2 = ["tj0r", "tj1r", "tj2r", "tj3r", "fGB"]

        input1 = np.array([[row[feature] for feature in features1] for row in self.data])

        input2 = np.array([[row[feature] for feature in features2] for row in self.data])

        graph = [row["graph"]  for row in self.data]
        tj = [row["tj"] for row in self.data]

        return input1, input2, graph, tj




