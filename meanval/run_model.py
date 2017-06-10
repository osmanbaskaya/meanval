from models.predict import SimpleEncoderModel
from preprocess import prepare_data


def run():
    model = SimpleEncoderModel()
    batch_size = 256
    training_iterator, vocab_size = prepare_data(dataset_type='train', batch_size=batch_size)
    validation_iterator, vocab_size = prepare_data(dataset_type='validate', batch_size=batch_size)
    model.create_model(vocab_size)
    model.run(training_iterator, validation_iterator)


if __name__ == '__main__':
    run()
