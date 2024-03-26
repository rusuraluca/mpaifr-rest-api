import os
import matplotlib.pyplot as plt

# Define age groups and their labels
age_groups = [(0, 12), (13, 18), (19, 25), (26, 35), (36, 45), (46, 55), (56, 65), (66, float('inf'))]
age_group_labels = ['0-12', '13-18', '19-25', '26-35', '36-45', '46-55', '56-65', '>=66']

# Initialize stats dictionary
stats = {'agedb_training': {label: {'m': 0, 'f': 0} for label in age_group_labels},
         'agedb_validation': {label: {'m': 0, 'f': 0} for label in age_group_labels},
         'agedb': {label: {'m': 0, 'f': 0} for label in age_group_labels}}

def process_directory(directory):
    for person_folder in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, person_folder)):
            continue  # Skip files, process only directories

        # Extract gender from folder name
        _, gender = person_folder.rsplit('_', 1)

        for image_file in os.listdir(os.path.join(directory, person_folder)):
            # Extract age from the file name
            age = int(image_file.split('_')[-1].split('.')[0])

            # Determine the age group
            for (start, end), label in zip(age_groups, age_group_labels):
                if start <= age <= end:
                    stats[directory][label][gender] += 1
                    break


# Process each directory
for dataset_type in ['agedb_training', 'agedb_validation', 'agedb']:
    process_directory(dataset_type)

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