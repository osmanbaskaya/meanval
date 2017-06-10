import data
import sys


def run():
    dataset = sys.argv[1]
    preprocessor = data.WMT15DataPreprocessor()
    print("Pre-processing has been started for {}".format(dataset), file=sys.stderr)
    preprocessor.preprocess(dataset)

if __name__ == '__main__':
    run()
