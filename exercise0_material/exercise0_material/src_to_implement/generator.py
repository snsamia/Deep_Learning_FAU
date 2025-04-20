import os
import json
import numpy as np
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        
        self.sample_count = len(self.labels)
        self.epoch_count = 0
        self.current_index = 0

        
        if self.sample_count < self.batch_size or self.batch_size == 0:
            self.batch_size = self.sample_count

        
        self.indices = np.arange(self.sample_count)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        
        if self.current_index * self.batch_size >= self.sample_count:
            self.epoch_count += 1
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indices)

        
        images = np.zeros((self.batch_size, *self.image_size), dtype=np.float32)
        labels = np.zeros(self.batch_size, dtype=int)

        start_idx = self.current_index * self.batch_size
        idx = 0
        while idx < self.batch_size and (start_idx + idx) < self.sample_count:
            img_idx = self.indices[start_idx + idx]
            img = np.load(os.path.join(self.file_path, f"{img_idx}.npy"))
            img = self.augment(img)
            images[idx] = img
            labels[idx] = self.labels[str(img_idx)]
            idx += 1

        
        if idx < self.batch_size:
            remaining = self.batch_size - idx
            for i in range(remaining):
                img_idx = self.indices[i]
                img = np.load(os.path.join(self.file_path, f"{img_idx}.npy"))
                images[idx + i] = self.augment(img)
                labels[idx + i] = self.labels[str(img_idx)]

        self.current_index += 1
        return images.copy(), labels.copy()

    def augment(self, img):
       
        if img.shape != tuple(self.image_size):
            img = np.resize(img, self.image_size)
        
        
        if self.mirroring:
            if np.random.choice([True, False]):
                img = np.fliplr(img)
            if np.random.choice([True, False]):
                img = np.flipud(img)
        
        
        if self.rotation:
            rotations = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
            img = np.rot90(img, rotations)
        
        return img

    def current_epoch(self):
        
        return self.epoch_count

    def class_name(self, int_label):
        
        class_map = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
        return class_map[int_label]

    def show(self):
        
        images, labels = self.next()
        fig = plt.figure(figsize=(12, 12))
        total_images = min(self.batch_size, len(images))
        columns = 3
        rows = (total_images // columns) + (1 if total_images % columns else 0)
        
        for i in range(1, total_images + 1):
            ax = fig.add_subplot(rows, columns, i)
            ax.imshow(images[i - 1].astype('uint8'))
            ax.set_title(self.class_name(labels[i - 1]))
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()





      