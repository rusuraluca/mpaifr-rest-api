import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from utils.data_utils import ImageFolderWithAgeGender


class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, labels, ages, genders):
        pass


class AgeGenderDistributionAnalysis(AnalysisStrategy):
    def analyze(self, labels, ages, genders):
        age_gender_groups = {
            0: {'Label': '0-12', 'Male': 0, 'Female': 0},
            1: {'Label': '13-18', 'Male': 0, 'Female': 0},
            2: {'Label': '19-25', 'Male': 0, 'Female': 0},
            3: {'Label': '26-35', 'Male': 0, 'Female': 0},
            4: {'Label': '36-45', 'Male': 0, 'Female': 0},
            5: {'Label': '46-55', 'Male': 0, 'Female': 0},
            6: {'Label': '56-65', 'Male': 0, 'Female': 0},
            7: {'Label': '>=65', 'Male': 0, 'Female': 0}
        }

        for age, gender in zip(ages, genders):
            if gender == 0:
                age_gender_groups[age]['Male'] += 1
            else:
                age_gender_groups[age]['Female'] += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        age_groups = [group['Label'] for group in age_gender_groups.values()]
        num_age_groups = len(age_groups)
        bar_width = 0.35
        index = np.arange(num_age_groups)

        male_counts = [group['Male'] for group in age_gender_groups.values()]
        female_counts = [group['Female'] for group in age_gender_groups.values()]

        ax.bar(index, male_counts, bar_width, label='Male', color='dodgerblue')
        ax.bar(index + bar_width, female_counts, bar_width, label='Female', color='mediumpurple')

        ax.set_xlabel('Age Group')
        ax.set_ylabel('Image Count')
        ax.set_title('Age-Gender Distribution')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(age_groups)
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class GenderDistributionAnalysis(AnalysisStrategy):
    def analyze(self, labels, ages, genders):
        male_count = sum(g == 0 for g in genders)
        female_count = sum(g == 1 for g in genders)
        total_count = male_count + female_count

        if total_count == 0:
            print("No data available for analysis.")
            return

        male_percentage = (male_count / total_count) * 100
        female_percentage = (female_count / total_count) * 100

        labels = 'Male', 'Female'
        sizes = [male_percentage, female_percentage]
        colors = ['dodgerblue', 'mediumpurple']
        explode = (0.1, 0)

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title(f'Gender Distribution: Male ({male_count}) vs. Female ({female_count})')
        plt.axis('equal')
        plt.show()


class AgeDistributionAnalysis(AnalysisStrategy):
    def analyze(self, labels, ages, genders):
        age_categories = {
            '0-12': 0,
            '13-18': 0,
            '19-25': 0,
            '26-35': 0,
            '36-45': 0,
            '46-55': 0,
            '56-65': 0,
            '65+': 0
        }

        for age in ages:
            if age == 0:
                age_categories['0-12'] += 1
            elif age == 1:
                age_categories['13-18'] += 1
            elif age == 2:
                age_categories['19-25'] += 1
            elif age == 3:
                age_categories['26-35'] += 1
            elif age == 4:
                age_categories['36-45'] += 1
            elif age == 5:
                age_categories['46-55'] += 1
            elif age == 6:
                age_categories['56-65'] += 1
            else:
                age_categories['65+'] += 1

        labels = list(age_categories.keys())
        sizes = list(age_categories.values())
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'orange', 'lightred', 'mediumpurple', 'dodgerblue']

        labels_with_data = [label for label, size in zip(labels, sizes) if size > 0]
        sizes_with_data = [size for size in sizes if size > 0]
        colors_with_data = [colors[i] for i, size in enumerate(sizes) if size > 0]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes_with_data, labels=labels_with_data, colors=colors_with_data,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Age Distribution')
        plt.axis('equal')
        plt.show()


class DatasetAnalysisContext:
    def __init__(self,
                 strategy: AnalysisStrategy,
                 dataset_path: str,
                 pattern=r'_|\.',
                 position_age=1,
                 position_gender=1,
                 cutoffs=None):
        if cutoffs is None:
            cutoffs = [12, 18, 25, 35, 45, 55, 65]
        self.strategy = strategy
        self.dataset_path = dataset_path
        self.pattern = pattern
        self.position_age = position_age
        self.position_gender = position_gender
        self.cutoffs = cutoffs
        self.transforms = transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def set_strategy(self, strategy: AnalysisStrategy):
        self.strategy = strategy

    def perform_analysis(self):
        dataset = ImageFolderWithAgeGender(root=self.dataset_path, pattern=self.pattern,
                                           position_age=self.position_age, position_gender=self.position_gender,
                                           cutoffs=self.cutoffs, transform=self.transforms)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        ages = []
        genders = []
        labels = []
        for _, label, age_group, gender_group in loader:
            ages.extend(age_group.numpy())
            genders.extend(gender_group.numpy())
            labels.extend(label.numpy())
        self.strategy.analyze(labels, ages, genders)


if __name__ == "__main__":
    path = '../data_training'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
    context.set_strategy(GenderDistributionAnalysis())
    context.perform_analysis()
    context.set_strategy(AgeDistributionAnalysis())
    context.perform_analysis()

    path = '../data_validation'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
    context.set_strategy(GenderDistributionAnalysis())
    context.perform_analysis()
    context.set_strategy(AgeDistributionAnalysis())
    context.perform_analysis()
