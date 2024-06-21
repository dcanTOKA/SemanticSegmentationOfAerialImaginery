import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from enums.label import Label
from models.unet import UNet
from utils.transform import get_transforms


class InferenceService:
    def __init__(self, model_path, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(3, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = get_transforms()

    def predict(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            prediction = torch.argmax(output, dim=1)
            return prediction.squeeze(0).cpu().numpy()

    @staticmethod
    def decode_segmentation(prediction):
        color_map = [label.hex_to_rgb() for label in Label]
        r = np.zeros_like(prediction).astype(np.uint8)
        g = np.zeros_like(prediction).astype(np.uint8)
        b = np.zeros_like(prediction).astype(np.uint8)

        for idx, color in enumerate(color_map):
            r[prediction == idx] = color[0]
            g[prediction == idx] = color[1]
            b[prediction == idx] = color[2]

        rgb = np.stack([r, g, b], axis=2)
        return Image.fromarray(rgb)

    def process_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                 f.endswith('.png') or f.endswith('.jpg')]

        for file_path in tqdm(files):
            image = Image.open(file_path)
            prediction = self.predict(image)
            segmented_image = self.decode_segmentation(prediction)
            segmented_image.save(os.path.join(output_folder, os.path.basename(file_path)))


# Example usage
if __name__ == '__main__':
    service = InferenceService('../results/1718058329/models/best_model.pth', num_classes=6)
    service.process_folder('/input', '/output')
