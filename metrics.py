# Utilities to determine metrics and performance

# Calculate the output of testing the neural network
def calculate_metrics(truth, predicted):
    # TODO
    pass

# Interpret output of neural network
def interpret_output(response):
    # TODO
    pass

# Calculate intersection over union confidence scores for bounding boxes
def iou(box1, box2):
    # TODO
    pass

# box_vals: (1500,7,7,8)
#   pc,bx,by,bw,bh,c1,c2,c3
def draw_box(box_vals):
    for img in box_vals:
        for cell in img:
            for box in cell:
                if box[0] != 0:
                    # Start point: bx - bw/2, by - bh/2
                    # End point: bx + bw/2, by + bh/2
                    xmin = box[1] - box[3] // 2
                    xmax = box[1] + box[3] // 2
                    ymin = box[2] - box[4] // 2
                    ymax = box[2] + box[4] // 2

                    start_p = (int(xmin), int(ymin))
                    end_p = (int(xmax), int(ymax))

                    # Red color, Line thickness of 3 px , font scale is 1
                    color = (0,0,255)
                    thickness = 3
                    fontScale = 1

                    # Lable the box
                    label_str = ''
                    if box[5] == 1:
                        label_str = 'circle'
                    if box[6] == 1:
                        label_str = 'square'
                    if box[7] == 1:
                        label_str = 'hexagon'

                    cv2.rectangle(cell, start_p, end_p, color, thickness)
                    cv2.putText(cell, label_str, (int(xmin), int(ymin) - 13), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
