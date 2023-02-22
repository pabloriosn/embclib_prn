from PIL import Image
import glob
import numpy as np

from class_emb import embclip


def main():
    model = embclip()
    actions = []

    images = sorted(glob.glob("sample/*43.png"))
    goal = 0

    for i in images:
        img = Image.open(i)
        im2arr = np.array(img)

        action = model.train(im2arr, goal)

        actions.append(action)

    print(actions)
    del()


if __name__ == "__main__":
    main()
