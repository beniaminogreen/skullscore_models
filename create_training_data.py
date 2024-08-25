from PIL import Image, ImageDraw, ImageFilter
import random
import tqdm
import glob

from utils import find_coeffs, add_salt_and_pepper, add_gauss, random_crop


training_images = [
 Image.open('originals/peasant.png'),
 Image.open('originals/bandit.png'),
 Image.open('originals/preist.png'),
 Image.open('originals/romantic.png'),
 Image.open('originals/royal/royal.png')
        ]


training_images = [(i, img.convert("RGBA")) for i, img in enumerate(training_images)]

def random_planar_transform(img, mask, z = .25):
    img = img.copy()
    mask = mask.copy()

    w,h = img.size
    pa = [(0,0), (w,0), (w,h), (0,h)]
    pb = [
            (random.uniform(0, w*z),0),
            (w,random.uniform(0, h*z)),
            (random.uniform(w*(1-z), w),h),
            (0,random.uniform(h, h*(1-z)))]

    coefs = find_coeffs(pb, pa)

    img = img.transform((w, h), Image.PERSPECTIVE, coefs, Image.BICUBIC)
    mask = mask.transform((w, h), Image.PERSPECTIVE, coefs, Image.BICUBIC)

    return (img, mask, pb)


# this handles planar tilt. Now we also need:
# 1) reporting position of image
# 2) rotation
# 3) resizing
def random_paste(backgrounds, training_examples):
    background = random.choice(backgrounds).copy()
    bg_width, bg_height = background.size

    annotations = []
    for _ in range(3):
        x = random.randint(0, int(bg_width*.65))
        y = random.randint(0, int(bg_height*.65))

        i, card = random.choice(training_examples)
        card = card.copy()
        width, height = card.size
        ratio = random.uniform(.4, .8)
        rotate = random.uniform(0, 360)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize the image
        card = card.resize((new_width, new_height))
        card = random_crop(card)

        mask = Image.new('L', card.size, 255)

        card, mask, corners = random_planar_transform(card, mask)

        card = card.rotate(rotate, expand=True)
        mask = mask.rotate(rotate, expand=True)

        background.paste(card, (x,y), mask)

        background = add_gauss(background, random.uniform(10,50))
        background = add_salt_and_pepper(background, random.uniform(0,.25))

        # draw = ImageDraw.Draw(background, "RGBA")
        # draw.rectangle(((x, y), (x+card.size[0], y+card.size[1])), outline=(0, 255, 0, 127))

        center_x = (x + (card.size[0]/2))/bg_width
        center_y = (y + (card.size[1]/2))/bg_height
        width = (card.size[0])/bg_width
        height =(card.size[1])/bg_height
        annotations.append((str(i), str(center_x), str(center_y), str(width), str(height)))

    return(background, annotations)


backgrounds = [Image.open(img_file) for img_file in glob.glob("backgrounds/*")]

def create_dataset(directory, n =2000):
    for i in tqdm.tqdm(range(n)):
        img, annotations = random_paste(backgrounds, training_images)
        img.save(f"datasets/{directory}/{i}.png")

        with open(f"datasets/{directory}/{i}.txt", "w") as f:
                f.writelines([" ".join(card) + "\n" for card in annotations])


create_dataset("training_images")
create_dataset("val_images")


