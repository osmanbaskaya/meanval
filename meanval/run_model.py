from models.predict import SimpleEncoderModel
from preprocess import prepare_data
from representation import get_embedding_weights


def run():
    model = SimpleEncoderModel()
    batch_size = 256
    training_iterator, vocabulary = prepare_data(dataset_type='train', batch_size=batch_size)
    validation_iterator, _ = prepare_data(dataset_type='validate', batch_size=batch_size)
    embedding_weights = get_embedding_weights(vocabulary)
    model.create_model(vocabulary, embedding_weights=embedding_weights)
    model.run(training_iterator, validation_iterator)


if __name__ == '__main__':
    run()
