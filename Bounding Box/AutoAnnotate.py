import cv2 as cv
import os
import pandas

# global b, c, d, e
pred_folder = r'E:\retina_dataset\predictions_0.8'
img_folder = r'E:\retina_dataset\images'
out_dir = r'E:\retina_dataset\Annotated_images'
i = 0
for filename in os.listdir(pred_folder):
    try:
        txt = os.path.join(pred_folder, filename)
        # since images are named after their annotation files
        img_name = filename[:-4]
        f = open(txt)
        df = pandas.read_table(f, delim_whitespace=True, names=('A', 'B', 'C', 'D', 'E', 'F'))
        x1 = df.iloc[:, 2]
        b = int(x1[0])
        print(b)
        y1 = df.iloc[:, 3]
        c = int(y1[0])
        print(c)
        x2 = df.iloc[:, 4]
        d = int(x2[0])
        print(d)
        y2 = df.iloc[:, 5]
        e = int(y2[0])
        print(e)
        # open image in image folder
        p = os.path.join(img_folder, img_name + ".jpg")
        image = cv.imread(p)
        startpoint = (c, b)
        endpoint = (e, d)
        colour = (0, 255, 0)
        thickness = 3
        # Draw a box with the coordinates
        img_mod = cv.rectangle(image, startpoint, endpoint, colour, thickness)
        # save the modified image
        cv.imwrite(os.path.join(out_dir, f"{img_name}_{i}.jpg"), img_mod)
        i += 1

    except:
        print("folder not selected")
