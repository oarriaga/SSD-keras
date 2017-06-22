try:
    import cv2
except ImportError:
    cv2 = None

import matplotlib.pyplot as plt
import numpy as np

def draw_boxes(box_data, original_image_array, arg_to_class, colors, font):
    if len(box_data) == 0:
        return
    x_min = box_data[:, 0]
    y_min = box_data[:, 1]
    x_max = box_data[:, 2]
    y_max = box_data[:, 3]
    classes = box_data[:, 4:]
    num_boxes = len(box_data)
    for box_arg in range(num_boxes):
        x_min_box = int(x_min[box_arg])
        y_min_box = int(y_min[box_arg])
        x_max_box = int(x_max[box_arg])
        y_max_box = int(y_max[box_arg])
        box_class_scores = classes[box_arg]
        label_arg = np.argmax(box_class_scores)
        score = box_class_scores[label_arg]
        class_name = arg_to_class[label_arg]
        color = colors[label_arg]
        display_text = '{:0.2f}, {}'.format(score, class_name)
        cv2.rectangle(original_image_array, (x_min_box, y_min_box),
                                (x_max_box, y_max_box), color, 2)
        cv2.putText(original_image_array, display_text,
                    (x_min_box, y_min_box - 30), font,
                    .7, color, 1, cv2.LINE_AA)

def plot_images(image_1, image_2, title_1='original image',
                            title_2='transformed image'):
    plt.figure(1)
    plt.subplot(121)
    plt.title(title_1)
    plt.imshow(image_1.astype('uint8'))
    plt.subplot(122)
    plt.title(title_2)
    plt.imshow(image_2.astype('uint8'))
    plt.show()


