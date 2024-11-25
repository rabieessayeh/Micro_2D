from Data import Data
from Model import MT_Model
from Trainer import Trainer
from paths import data_folder, dataset_folder

if __name__ == '__main__':
    data = Data(data_folder, dataset_folder, normalisation_method="min_max") #standard_score
    #data.prepare_data()
    #batch = data.get_dataset(3)

    model = MT_Model()
    trainer = Trainer(model=model, data=data, dataset_folder=dataset_folder, num_epochs=5)
    trainer.train()

