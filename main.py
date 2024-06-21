from config.training_config import TrainingConfig
from services.train import TrainService


def main():
    training_config = TrainingConfig()

    train_service = TrainService(training_config)

    train_service.train()

    print("Training Losses:", train_service.train_loss)
    print("Validation Losses:", train_service.val_loss)
    print("Test Loss:", train_service.test_loss)


if __name__ == "__main__":
    main()
