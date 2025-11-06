from PIL import Image
import os

class ImageProcessor:
    def __init__(self, image_path, tile_size):
        self.tile_size = tile_size  # Tamaño del cuadro en píxeles
        self.img = Image.open(image_path)

    def check_empty_tiles(self, img):
        img = img.convert('L')
        extrema = img.getextrema()
        return extrema == (255, 255)

    def get_grid(self, output_folder="uploads"):
        img_width, img_height = self.img.size

        os.makedirs(output_folder, exist_ok=True)

        count = 0
        for i in range(0, img_height, self.tile_size):
            for j in range(0, img_width, self.tile_size):
                box = (j, i, j + self.tile_size, i + self.tile_size)
                cropped_img = self.img.crop(box)
                if self.check_empty_tiles(cropped_img):
                    continue

                cropped_img.save(os.path.join(output_folder, f"cuadro{count}.png"))    
                count += 1
