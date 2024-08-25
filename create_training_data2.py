from PIL import Image, ImageDraw, ImageFilter
import random
import tqdm
import glob
import torch

from torchvision.transforms import v2
from torchvision import io, utils
from torchvision import tv_tensors

from utils import find_coeffs, add_salt_and_pepper, add_gauss, random_crop


training_images = [
 Image.open('originals/peasant.png'),
 Image.open('originals/bandit.png'),
 Image.open('originals/preist.png'),
 Image.open('originals/romantic.png'),
 Image.open('originals/royal/royal.png')
        ]

def randomly_transform_card(card, out_size = 500):
    card = card.copy()
    bbox = torch.tensor([5, 5, card.size[0]-5, card.size[1]-5])
    boxes = tv_tensors.BoundingBoxes(bbox, format="XYXY", canvas_size=(card.size[1], card.size[0]))

    transforms = v2.Compose([
        v2.RandomResize(40, 100),
        v2.CenterCrop((out_size, out_size)),
        v2.RandomAffine(180),
        v2.RandomPerspective(.75)
    ])

    img, bboxes = transforms(card, boxes)

    bbox = bboxes[0]

    return img, tuple(coord.item() for coord in bbox)

training_images = [(i, img.convert("RGBA")) for i, img in enumerate(training_images)]

backgrounds = [Image.open(file) for file in glob.iglob("backgrounds/*")]


color_transform = v2.Compose([
    v2.RandomGrayscale(p=0.1),  # Convert to grayscale with 20% probability
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Randomly adjust sharpness
    v2.RandomAutocontrast(p=0.5),  # Randomly apply autocontrast
    v2.RandomEqualize(p=0.5),   # Randomly apply histogram equalization
    v2.RandomChoice([
        v2.RandomChannelPermutation(),
        v2.GaussianBlur(3),
        v2.ElasticTransform(),
        v2.ColorJitter(brightness=.5, hue=.3),
        ]),
    v2.RandomInvert(p=.1),
])

def create_image(n = 5):
    yolo_labels = []
    background = random.choice(backgrounds).copy()
    width, height = background.size
    for j in range(n):
        label , card_img = random.choice(training_images)

        card, bbox = randomly_transform_card(card_img)

        x_off = int(random.uniform(-bbox[0], width-bbox[2]))
        y_off = int(random.uniform(-bbox[1], height-bbox[3]))

        background.paste(card, (x_off, y_off), card)

        # draw = ImageDraw.Draw(background, "RGBA")
        # draw.rectangle(
        #         ((x_off + bbox[0], y_off+bbox[1]),
        #         (x_off + bbox[2], y_off+bbox[3])),
        #         outline=(0,255,0,255)
        #         )


        card_width = abs(bbox[2] - bbox[0])
        card_height = abs(bbox[3] - bbox[1])

        x1 = max(x_off + bbox[0] + (card_width/2),0)
        y1 = max(y_off + bbox[1] + (card_height/2),0)

        yolo_labels.append((
            str(label),
            str(x1 / width),
            str(y1 / height),
            str(card_width / width),
            str(card_height / height)
            ))

    background = color_transform(background)
    background = add_gauss(background, random.uniform(0,15))
    background = add_salt_and_pepper(background, random.uniform(0,.25))

    return (background, yolo_labels)


def create_dataset(directory, n =2000):
    for i in tqdm.tqdm(range(n)):
        img, annotations = create_image(3)
        img.save(f"datasets/{directory}/{i}.png")

        with open(f"datasets/{directory}/{i}.txt", "w") as f:
            f.writelines([" ".join(card) + "\n" for card in annotations])



create_dataset("training_images", n = 10000)
create_dataset("val_images", n = 500)
