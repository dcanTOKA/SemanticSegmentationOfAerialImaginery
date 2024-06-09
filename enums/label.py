from enum import Enum
import numpy as np

class Label(Enum):
    BUILDING = "#3C1098"
    LAND = "#8429F6"
    ROAD = "#6EC1E4"
    VEGETATION = "#FEDD3A"
    WATER = "#E2A929"
    UNLABELED = "#9B9B9B"

    def hex_to_rgb(self):
        hex_code = self.value.lstrip('#')
        return np.array(tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4)))

# print("Label: BUILDING, RGB Color:", Label.BUILDING.hex_to_rgb())
