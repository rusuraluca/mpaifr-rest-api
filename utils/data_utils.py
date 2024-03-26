import re
from os.path import basename
from torchvision.datasets import ImageFolder


def path2age(path, pat, pos):
    return int(re.split(pat, basename(path))[pos])


class ImageFolderWithAges(ImageFolder):
    """
    A custom dataset class that extends torchvision.datasets.ImageFolder to include
    age information extracted from image filenames.
    """
    def __init__(self, pattern, position, *args, **kwargs):
        super(ImageFolderWithAges, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.position = position

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        age = path2age(path, self.pattern, self.position)
        return img, label, age


class ImageFolderWithAgeGroup(ImageFolder):
    """
    Extends torchvision.datasets.ImageFolder to include age group information.
    Age groups are determined by a list of cutoff ages.
    """
    def __init__(self, pattern, position, cutoffs, *args, **kwargs):
        super(ImageFolderWithAgeGroup, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.position = position
        self.cutoffs = cutoffs

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        age = path2age(path, self.pattern, self.position)
        age_group = self.find_age_group(age)
        return img, label, age_group

    def find_age_group(self, age):
        age_group = next((i for i, cutoff in enumerate(self.cutoffs) if age <= cutoff), len(self.cutoffs))
        return age_group


class ImageFolderWithAgeGender(ImageFolder):
    def __init__(self, pattern, position_age, position_gender, cutoffs, *args, **kwargs):
        super(ImageFolderWithAgeGender, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.position_age = position_age
        self.position_gender = position_gender
        self.cutoffs = cutoffs

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        age = path2age(path, self.pattern, self.position_age)
        gender = self.path2gender(path, self.pattern, self.position_gender)
        age_group = self.find_age_group(age)
        return img, label, age_group, gender

    def path2gender(self, path, pat, pos):
        gender_str = re.split(pat, basename(path))[pos]
        return 0 if gender_str.lower().startswith('m') else 1

    def find_age_group(self, age):
        age_group = next((i for i, cutoff in enumerate(self.cutoffs) if age <= cutoff), len(self.cutoffs))
        return age_group
