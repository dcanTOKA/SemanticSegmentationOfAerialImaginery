import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from config.training_config import TrainingConfig
from enums.label import Label


class MetricsService:
    def __init__(self, training_config):
        self.training_config: TrainingConfig = training_config
        self.results_dir = os.path.join(self.training_config.results_dir, f"{str(int(time.time()))}")
        os.makedirs(self.results_dir, exist_ok=True)
        self.models_dir = os.path.join(self.results_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.reports_dir = os.path.join(self.results_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.cm_dir = os.path.join(self.results_dir, "confusion_matrix")
        os.makedirs(self.cm_dir, exist_ok=True)
        self.iou_dir = os.path.join(self.results_dir, "iou")
        os.makedirs(self.iou_dir, exist_ok=True)

    def calculate_iou(self, preds, labels, smooth=1e-6):
        num_classes = self.training_config.num_classes
        ious = []

        for cls in range(num_classes):
            pred_indices = (preds == cls)
            target_indices = (labels == cls)
            intersection = (pred_indices & target_indices).sum()
            union = (pred_indices | target_indices).sum()
            iou = (intersection + smooth) / (union + smooth)
            ious.append(iou.item())

        mean_iou = sum(ious) / len(ious)
        return mean_iou, ious

    def calculate_metrics(self, preds, labels):
        report = classification_report(labels, preds, output_dict=True, labels=Label.get_numeric_labels(),
                                       target_names=[label.name for label in Label], zero_division=0)
        conf_matrix = confusion_matrix(labels, preds, labels=Label.get_numeric_labels(), normalize='true')

        all_preds_tensor = torch.tensor(preds)
        all_labels_tensor = torch.tensor(labels)

        mIoU, ious = self.calculate_iou(all_preds_tensor, all_labels_tensor)

        return report, conf_matrix, mIoU, ious

    def save_metrics(self, epoch, report, conf_matrix, ious, mean_iou, mode="val"):
        if mode == "val":
            report_filename = os.path.join(self.reports_dir,
                                           f"{mode}_classification_report_epoch_{epoch + 1}.csv")
            conf_matrix_filename = os.path.join(self.cm_dir,
                                                f"{mode}_confusion_matrix_epoch_{epoch + 1}.csv")
            iou_filename = os.path.join(self.iou_dir, f"{mode}_iou_epoch_{epoch + 1}.csv")

            cm_fig_name = f"{mode}_confusion_matrix_epoch_{epoch + 1}.png"
            report_fig_name = f"{mode}_report_epoch_{epoch + 1}.png"
            iou_fig_name = f"{mode}_iou_epoch_{epoch + 1}.png"

        elif mode == "test":
            report_filename = os.path.join(self.reports_dir,
                                           f"{mode}_classification_report.csv")
            conf_matrix_filename = os.path.join(self.cm_dir,
                                                f"{mode}_confusion_matrix.csv")
            iou_filename = os.path.join(self.iou_dir, f"{mode}_iou.csv")

            cm_fig_name = f"{mode}_confusion_matrix.png"
            report_fig_name = f"{mode}_report.png"
            iou_fig_name = f"{mode}_iou.png"

        else:
            raise ValueError("Invalid mode operation")

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(report_filename)

        conf_matrix_df = pd.DataFrame(conf_matrix,
                                      index=[f'A {i}' for i in [label.name for label in Label]],
                                      columns=[f'P {i}' for i in [label.name for label in Label]])
        conf_matrix_df.to_csv(conf_matrix_filename)

        iou_df = pd.DataFrame({'Class': [label.name for label in Label], 'IoU': ious})
        iou_df.to_csv(iou_filename, index=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Class', y='IoU', data=iou_df)
        plt.title(f'IoU by Class ({mode.capitalize()})')
        plt.xlabel('Classes')
        plt.ylabel('IoU')
        plt.savefig(os.path.join(self.iou_dir, iou_fig_name))
        plt.close()

        with open(iou_filename, 'a') as file:
            file.write(f"\nMean IoU,{mean_iou}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_df, annot=True, fmt=".2f", cmap="Blues", cbar=False)
        plt.title(f'Confusion Matrix ({mode.capitalize()})')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(os.path.join(self.cm_dir, cm_fig_name))
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(report_df).iloc[:-1, :].T, annot=True)
        plt.title(f'Classification Report ({mode.capitalize()})')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.savefig(os.path.join(self.reports_dir, report_fig_name))
        plt.close()

    def save_results(self, epoch, train_loss, train_acc, val_loss, val_acc, mode="epoch"):
        filename = f"{mode}_results.txt"
        with open(os.path.join(self.results_dir, filename), 'a') as file:
            file.write(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

    def save_model(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.models_dir, filename))
