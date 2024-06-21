import datetime

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.training_config import TrainingConfig
from models.unet import UNet
from models.unet_3_plus import Unet3Plus
from utils.logvisor import logger
from utils.transform import get_transforms
from .dataloader import SegmentationDataset
from .metrics_service import MetricsService


class TrainService:
    def __init__(self, training_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_config: TrainingConfig = training_config
        self.metrics_service: MetricsService = MetricsService(training_config)
        self.model: nn.Module = UNet(3, self.training_config.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_config.learning_rate)
        self.criteria = nn.CrossEntropyLoss()
        self.transform = get_transforms()
        self.train_dataset: SegmentationDataset = SegmentationDataset(root_dir=self.training_config.root_dir,
                                                                      subset='train',
                                                                      image_transform=self.transform,
                                                                      mask_transform=None)

        self.val_dataset: SegmentationDataset = SegmentationDataset(root_dir=self.training_config.root_dir,
                                                                    subset='val',
                                                                    image_transform=self.transform,
                                                                    mask_transform=None)

        self.test_dataset: SegmentationDataset = SegmentationDataset(root_dir=self.training_config.root_dir,
                                                                     subset='test',
                                                                     image_transform=self.transform,
                                                                     mask_transform=None)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.training_config.batch_size, shuffle=True,
                                       num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.training_config.batch_size, shuffle=False,
                                     num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.training_config.batch_size, shuffle=False,
                                      num_workers=2)

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0

        loop = tqdm(self.train_loader, leave=True)
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)  # torch.Size([batch_size, num_classes, H, W])

            if type(outputs) != list:
                loss = self.criteria(outputs, masks)
                preds = torch.argmax(outputs, dim=1)
            else:
                loss = 0
                for output in outputs:
                    loss += self.criteria(output, masks)

                preds = torch.argmax(outputs[0], dim=1)

            acc = torch.tensor(torch.sum(masks == preds).item() / preds.numel())
            total_acc += acc.item()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch + 1}/{self.training_config.num_epochs}]")
            loop.set_postfix(loss=loss.item())
            loop.update()

        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)

    def validation(self, epoch=None, mode="val"):
        if mode == "test":
            data_loader = self.test_loader
            desc = "Testing stage"
        elif mode == "val":
            data_loader = self.val_loader
            desc = "Val"  # dummy
        else:
            raise ValueError("Invalid mode for validation or test operation")

        self.model.eval()

        total_loss = 0
        total_acc = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            loop = tqdm(data_loader, leave=False)
            for images, masks in data_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                if type(outputs) != list:
                    loss = self.criteria(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)
                else:
                    loss = 0
                    for output in outputs:
                        loss += self.criteria(output, masks)

                    preds = torch.argmax(outputs[0], dim=1)

                acc = torch.tensor(torch.sum(masks == preds).item() / preds.numel())

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())

                if mode == "val":
                    loop.set_description(f"Val Epoch [{epoch + 1}/{self.training_config.num_epochs}]")
                else:
                    loop.set_description(desc)
                loop.set_postfix(loss=loss.item())
                loop.update()

                total_loss += loss.item()
                total_acc += acc.item()

        report, conf_matrix, mean_iou, ious = self.metrics_service.calculate_metrics(all_preds, all_labels)
        self.metrics_service.save_metrics(epoch, report, conf_matrix, ious, mean_iou, mode)

        return total_loss / len(data_loader), total_acc / len(data_loader)

    def train(self):

        logger.info("###########################################################")
        logger.info(f"Training is starting : {datetime.datetime.now()}")
        logger.info(f"Device : {self.device}")
        logger.info("###########################################################")

        for epoch in range(self.training_config.num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validation(epoch)

            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            self.metrics_service.save_results(epoch, train_loss, train_acc, val_loss, val_acc)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.metrics_service.save_model(self.model, "best_model.pth")
            logger.info(
                f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        self.metrics_service.save_model(self.model, "last_model.pth")
        test_loss, test_accuracy = self.validation(mode="test")
        logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
