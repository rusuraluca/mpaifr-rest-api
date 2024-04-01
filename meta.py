age_cutoffs = [12, 18, 25, 35, 45, 55, 65]
genders = [0, 1]

AGEDB = {
    'training_root': 'agedb_training/',
    'validation_root': 'agedb_training/',
    'save_root': 'results/model/17/',
    'pattern': r'_|\.',
    'position_age': 1,
    'position_gender': 1,
    'num_classes': 572
}

WANDB = {
    'project': 'mpaifr',
    'entity': 'ralucarusu',
    'learning_rate': 0.01,
    'epochs': 10,
    'batch_size': 128,
    'loss_head': 'cosface',
    'embedding_dimension': 512,
    'model': 'results/model/16/model_epoch_9.pth',
    'save_model': 'results/model/17/model_epoch_9.pth',
    'lambdas': (0.1, 0.1, 0.1),
}
