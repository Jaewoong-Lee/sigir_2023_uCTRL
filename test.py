from recbole.config import Config
from recbole.data import create_dataset, data_preparation

if __name__ == '__main__':
    config = Config(model='BPR', dataset='amazon-book', config_file_list=['recbole/properties/dataset/amazon-book.yaml'])
    dataset = create_dataset(config)


    train_data, valid_data, test_data = data_preparation(config, dataset)