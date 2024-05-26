from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(
        np.expand_dims(a[:, 0], 1), b[:, 0]
    )
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(
        np.expand_dims(a[:, 1], 1), b[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = (
            np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1)
            + area
            - iw * ih
    )

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, num_classes=11, score_threshold=0.05, max_detections=100):
    """Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [
        [None for i in range(num_classes)] for j in range(len(dataset))
    ]

    retinanet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]

            # run network
            scores, labels, boxes = retinanet(
                data["image"].to(device).float().unsqueeze(dim=0)
            )
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [
                        image_boxes,
                        np.expand_dims(image_scores, axis=1),
                        np.expand_dims(image_labels, axis=1),
                    ],
                    axis=1,
                )

                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = np.zeros((0, 5))

            print("{}/{}".format(index + 1, len(dataset)), end="\r")

    return all_detections


def _get_annotations(val_dataset, num_classes=11):
    """Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [
        [None for i in range(num_classes)] for j in range(len(val_dataset))
    ]

    for i in range(len(val_dataset)):
        # load the annotations
        annotations = val_dataset[i]['target']['boxes']

        # copy detections to all_annotations
        for label in range(num_classes):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].cpu().numpy().copy()

        print("{}/{}".format(i + 1, len(val_dataset)), end="\r")

    return all_annotations


class BBoxTransform(nn.Module):
    """[summary]"""

    def __init__(self, mean=None, std=None):
        """[summary]

        Args:
            mean ([type], optional): [description]. Defaults to None.
            std ([type], optional): [description]. Defaults to None.
        """
        super(BBoxTransform, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).to(
                device
            )
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            ).to(device)
        else:
            self.std = std

    def forward(self, boxes, deltas):
        """computes the output coordinates

        Args:
            boxes ([tensor]): [description]
            deltas ([tensor]): [description]

        Returns:
            [tensor]: [description]
        """

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes


class ClipBoxes(nn.Module):
    """[summary]"""

    def __init__(self):
        """[summary]"""
        super(ClipBoxes, self).__init__()

    @staticmethod
    def forward(boxes, img):
        """[clips the box coordinates]

        Args:
            boxes ([tensor]): [bounding box coordinates]
            img ([tensor]): [image tensor]

        Returns:
            [type]: [description]
        """
        # batch_size, num_channels, height, width = img.shape
        _, _, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
