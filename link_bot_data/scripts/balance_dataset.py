import argparse

from link_bot_models.multi_environment_datasets import MultiEnvironmentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")

    args = parser.parse_args()

    dataset = MultiEnvironmentDataset.load_dataset(args.dataset)
    generator = dataset.generator()
    positive_count = 0
    negative_count = 0
    for i in range(len(generator)):
        x, y = generator[i]
        if y['combined']


if __name__ == '__main__':
    main()
