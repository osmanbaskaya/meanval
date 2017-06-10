from models.predict import SimpleEncoderModel
from preprocess import prepare_data


def run():
    model = SimpleEncoderModel()
    data_iterator, vocab_size = prepare_data(dataset_type='train', batch_size=64)
    model.create_model(vocab_size)
    model.run(data_iterator)


if __name__ == '__main__':
    run()
