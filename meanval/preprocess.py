import data


def run():
    preprocessor = data.WMT15DataPreprocessor()
    preprocessor.preprocess('wmt15.all-en.csv')

if __name__ == '__main__':
    run()
