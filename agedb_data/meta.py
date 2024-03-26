age_cutoffs = [12, 18, 25, 35, 45, 55, 65]
gender = {'M': 0, 'F': 1}

AGEDB = {
    'training_root': 'agedb_training/',
    'validation_root': 'agedb_validation/',
    'save_root': 'results/model/1/',
    'pattern': r'_|\.',
    'position_age': 1,
    'position_gender': 1,
    'num_classes': 1000
}

WANDB = {
    'project': 'mpaifr',
    'entity': 'ralucarusu',
    'learning_rate': 0.1,
    'epochs': 5,
    'batch_size': 512,
    'loss_head': 'cosface',
    'embedding_dimension': 512,
    'model': 'results/model/1/model_epoch_3.pth',
    'lambdas': (0.1, 0.1, 0.1),
}
