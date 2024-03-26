import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchvision import transforms

from dal import DAL
from utils.data_utils import ImageFolderWithAgeGender
from utils.train_utils import MetricsRecorder
from data.meta import age_cutoffs, AGEDB, WANDB
from itertools import chain
import wandb


class Training:
    """

    """
    def __init__(self):
        self.config = WANDB
        self.dataset = AGEDB

        wandb.init(project=self.config["project"], entity=self.config["entity"])
        wandb.config.update({
            "learning_rate": self.config["learning_rate"],
            "epochs": self.config["epochs"],
            "batch_size": self.config["batch_size"],
            "loss_head": self.config["loss_head"],
            "embedding_dimension": self.config["embedding_dimension"],
        })
        self.model = DAL(
            loss_head=self.config["loss_head"],
            num_classes=self.dataset["num_classes"],
            embedding_dimension=self.config["embedding_dimension"],
        )
        self.model.load_state_dict(torch.load(self.config["model"]))
        print(f'Loaded weights from {self.config["model"]}')

    def train(self):
        trainer = Trainer(
            self.model,
            self.dataset,
            learning_rate=self.config["learning_rate"],
            batch_size=self.config["batch_size"],
            lambdas=self.config["lambdas"]
        )
        wandb.watch(self.model, log="all")
        trainer.train(epochs=self.config["epochs"])

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(self.config["model"])
        wandb.log_artifact(artifact)
        wandb.finish()


class Trainer:
    """
    A class to encapsulate training logic for the DAL model, handling both training and validation phases.
    """

    def __init__(self, model, dataset, learning_rate=0.01, batch_size=512, lambdas=(0.1, 0.1, 0.1), print_freq=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lambdas = lambdas
        self.print_freq = print_freq

        self.optimizer = optim.SGD(
            params=chain(
                self.model.age_classifier.parameters(),
                self.model.gender_classifier.parameters(),
                self.model.RFM.parameters(),
                self.model.margin_loss.parameters(),
                self.model.backboneCNN.parameters(),
                self.model.DALR.parameters()
            ),
            lr=learning_rate,
            momentum=0.9
        )

        self.optimizer_DAL = optim.SGD(
            self.model.DALR.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

        self.transforms = transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainingDAL = False

        self.id_recorder = MetricsRecorder()
        self.age_recorder = MetricsRecorder()
        self.gender_recorder = MetricsRecorder()

    def set_train_mode(self, state):
        """
        Adjusts which parts of the model should be optimized.
        """
        self.trainingDAL = not state
        self.set_grads(self.model.RFM, state)
        self.set_grads(self.model.backboneCNN, state)
        self.set_grads(self.model.margin_loss, state)
        self.set_grads(self.model.age_classifier, state)
        self.set_grads(self.model.gender_classifier, state)
        self.set_grads(self.model.DALR, not state)

    @staticmethod
    def set_grads(mod, state):
        """
        Enables or disables gradients for a module.
        """
        for param in mod.parameters():
            param.requires_grad = state

    @staticmethod
    def flip_grads(mod):
        """
        Flips the gradients for adversarial training.
        """
        for param in mod.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = -param.grad

    def compute_loss(self, id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, dal_loss):
        total_loss = id_loss + self.lambdas[0] * age_loss + self.lambdas[1] * gender_loss + self.lambdas[2] * dal_loss
        return total_loss

    def run_epoch(self, loader, train):
        if loader is None:
            print("Warning: DataLoader is None. Skipping this epoch.")
            return

        phase = 'Training' if train else 'Validation'
        self.model.train() if train else self.model.eval()

        self.age_recorder.reset()
        self.gender_recorder.reset()
        self.id_recorder.reset()

        for i, (images, labels, age_groups, genders) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            age_groups = age_groups.to(self.device)
            genders = genders.to(self.device)
            id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, dal_loss \
                = self.model(images, labels, age_groups, genders)

            self.id_recorder.gulp(len(age_groups), id_loss.item(), id_accuracy.item())
            self.age_recorder.gulp(len(age_groups), age_loss.item(), age_accuracy.item())
            self.gender_recorder.gulp(len(genders), gender_loss.item(), gender_accuracy.item())

            loss = self.compute_loss(
                id_loss,
                id_accuracy,
                age_loss,
                age_accuracy,
                gender_loss,
                gender_accuracy,
                dal_loss
            )

            if train:
                if i % 70 == 0:
                    self.set_train_mode(True)  # maximize canonical correlation
                elif i % 70 == 20:
                    self.set_train_mode(False)  # minimize feature correlation

                if self.trainingDAL:
                    # feature correlation minimization
                    self.optimizer_DAL.zero_grad()
                    loss.backward()
                    self.flip_grads(self.model.DALR)
                    self.optimizer_DAL.step()
                else:
                    # canonical correlation maximization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if i % self.print_freq == 0:
                print(f"{phase} - Batch {i}/{len(loader)}: "
                      f"Total Loss {loss.item()}, "
                      f"Id Loss {id_loss.item():.4f}, Acc {id_accuracy.item():.4f}, "
                      f"Age Loss {age_loss.item():.4f}, Acc {age_accuracy.item():.4f}, "
                      f"Gender Loss {gender_loss.item():.4f}, Acc {gender_accuracy.item():.4f}, "
                      f"DAL Loss {dal_loss.item():.8f} \n"
                      )
            if not train:
                print(f"Avg Validation Meta: "
                      f"Acc Id {self.id_recorder.total_correct}\n")

        # average training meta after epoch
        print(f"Avg Training Meta: "
              f"Id {self.id_recorder.excrete().result()}, "
              f"Age {self.age_recorder.excrete().result()}, "
              f"Gender {self.gender_recorder.excrete().result()}\n"
              )

    def save_model(self, epoch):
        model_path = os.path.join(self.dataset['save_root'], 'model_epoch_{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def wandb_save_model(self, epoch):
        model_filename = f'model_epoch_{epoch}.pth'
        model_path = os.path.join(wandb.run.dir, model_filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        wandb.save(model_path)

    def load_data(self, train=True):
        path_key = 'training_root' if train else 'validation_root'
        dataset_path = self.dataset.get(path_key)

        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist.")
            return None

        dataset = ImageFolderWithAgeGender(
            pattern=self.dataset['pattern'],
            position_age=self.dataset['position_age'],
            position_gender=self.dataset['position_gender'],
            cutoffs=age_cutoffs,
            root=dataset_path,
            transform=self.transforms)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, epochs):
        training_loader = self.load_data(train=True)
        validation_loader = self.load_data(train=False)

        for epoch in range(epochs):
            self.run_epoch(training_loader, train=True)

            if validation_loader:
                with torch.no_grad():
                    self.run_epoch(validation_loader, train=False)

            self.save_model(epoch)
