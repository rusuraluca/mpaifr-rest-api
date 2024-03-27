import os
import matplotlib.pyplot as plt


class DatasetConfig(object):
    _instance = None
    _age_groups = None
    _age_group_labels = None
    _genders = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = super(DatasetConfig, cls).__new__(cls)
            cls._age_groups = [(0, 12), (13, 18), (19, 25), (26, 35), (36, 45), (46, 55), (56, 65), (66, float('inf'))]
            cls._age_group_labels = ['0-12', '13-18', '19-25', '26-35', '36-45', '46-55', '56-65', '>=66']
            cls._genders = ['m', 'f']
        return cls._instance

    @property
    def age_groups(self):
        return self._age_groups

    @property
    def age_group_labels(self):
        return self._age_group_labels

    @property
    def genders(self):
        return self._genders


class DatasetManager(object):
    _instance = None
    _stats = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = super(DatasetManager, cls).__new__(cls)
            dataset_config = DatasetConfig.instance()
            cls._stats = {
                'agedb_training': {label: {'m': 0, 'f': 0} for label in dataset_config.age_group_labels},
                'agedb_validation': {label: {'m': 0, 'f': 0} for label in dataset_config.age_group_labels},
                'agedb': {label: {'m': 0, 'f': 0} for label in dataset_config.age_group_labels}
            }
        return cls._instance

    @property
    def stats(self):
        return self._stats


class DatasetFactory:
    @staticmethod
    def process_directory(directory):
        dataset_manager = DatasetManager.instance()
        dataset_config = DatasetConfig.instance()

        for person_folder in os.listdir(directory):
            if not os.path.isdir(os.path.join(directory, person_folder)):
                continue

            _, gender = person_folder.rsplit('_', 1)
            if gender not in dataset_config.genders:
                continue  # Ensures gender is either 'm' or 'f'

            for image_file in os.listdir(os.path.join(directory, person_folder)):
                age = int(image_file.split('_')[-1].split('.')[0])

                for (start, end), label in zip(dataset_config.age_groups, dataset_config.age_group_labels):
                    if start <= age <= end:
                        dataset_key = directory.split('/')[-1]
                        dataset_manager.stats[dataset_key][label][gender] += 1
                        break

    @staticmethod
    def get_dataset(directory):
        DatasetFactory.process_directory(directory)


class AgeDistributionPlot:
    def __init__(self, stats):
        self.stats = stats

    def plot(self):
        age_group_labels = DatasetConfig.instance().age_group_labels
        total_counts_per_age_group = {label: 0 for label in age_group_labels}

        for dataset in self.stats.values():
            for age_group, counts in dataset.items():
                total_counts_per_age_group[age_group] += sum(counts.values())

        ages = list(total_counts_per_age_group.keys())
        counts = [total_counts_per_age_group[age_group] for age_group in ages]

        plt.figure(figsize=(10, 6))
        plt.bar(ages, counts, color='skyblue')
        plt.xlabel('Age Groups')
        plt.ylabel('Number of Individuals')
        plt.title('Age Distribution Across Datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class Statistics:
    def __init__(self, strategy):
        self._strategy = strategy

    def execute(self):
        self._strategy.plot()


dataset_manager = DatasetManager.instance()
DatasetFactory.get_dataset("agedb")
age_distribution_plot = AgeDistributionPlot(dataset_manager.stats)
statistics = Statistics(age_distribution_plot)
statistics.execute()


dataset_manager = DatasetManager.instance()
DatasetFactory.get_dataset("agedb_training")
age_distribution_plot = AgeDistributionPlot(dataset_manager.stats)
statistics = Statistics(age_distribution_plot)
statistics.execute()


dataset_manager = DatasetManager.instance()
DatasetFactory.get_dataset("agedb_validation")
age_distribution_plot = AgeDistributionPlot(dataset_manager.stats)
statistics = Statistics(age_distribution_plot)
statistics.execute()


"""


# Plotting the statistics
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for i, (dataset, data) in enumerate(stats.items()):
    males = [data[label]['m'] for label in age_group_labels]
    females = [data[label]['f'] for label in age_group_labels]

    if dataset != 'agedb':  # Skip non-agedb datasets for line plots
        axs[i].plot(age_group_labels, males, label='Male', marker='o')
        axs[i].plot(age_group_labels, females, label='Female', marker='o')
        axs[i].set_title(f'{dataset.capitalize()} Set')
        axs[i].set_ylabel('Number of Images')
        axs[i].set_xlabel('Age Groups')
    else:
        # For agedb, plot a histogram of the total counts
        total_counts = [males[j] + females[j] for j in range(len(age_group_labels))]
        axs[i].bar(age_group_labels, total_counts, color='skyblue')
        axs[i].set_title('Total Image Distribution in Agedb Dataset')
        axs[i].set_ylabel('Total Number of Images')

plt.xticks(rotation=45)
plt.xlabel('Age Groups')
plt.tight_layout()
plt.show()

print("Dataset visualization complete.")

"""