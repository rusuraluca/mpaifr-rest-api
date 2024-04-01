import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from utils.data_utils import ImageFolderWithAgeGender


class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, ages, genders):
        pass


class AgeGenderDistributionAnalysis(AnalysisStrategy):
    def analyze(self, ages, genders):
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
        ax.set_ylabel('Count')
        ax.set_title('Age-Gender Distribution')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(age_groups)
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
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
        for _, _, age_group, gender_group in loader:
            ages.extend(age_group.numpy())
            genders.extend(gender_group.numpy())
        self.strategy.analyze(ages, genders)


if __name__ == "__main__":
    path = 'agedb/'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
