import torch

def continuous_box_center(cx: torch.Tensor, cy: torch.Tensor, S: int):
    "assuming normalized inputs relative to image size"
    cell_x = (cx * S).floor().long()
    cell_y = (cy * S).floor().long()
    return cell_x, cell_y


def cell_offset(cx: torch.Tensor, cy: torch.Tensor):
    "assuming normalized inputs relative to image size"
    cx_s = cx * S
    cy_s = cy * S
    return cx_s - cx_s.floor(), cy_s - cy_s.floor()

def ground_truth_ij(gt_boxes: torch.Tensor):
    "normalized inputs, boxes as [batch, x, y, w, h, c]"
    cx, cy = gt_boxes[..., 1], gt_boxes[..., 2]
    cx, cy = continuous_box_center(cx, cy)
    return torch.cat([cx, cy, gt_boxes[..., -1]], dim = -1)



if __name__ == "__main__":
    BATCH, S, B, C = 32, 7, 2, 20
    prediction = torch.randn(size=(BATCH, S,S, (B*5 + C)))

    ground_truth = torch.tensor([
    [0.65, 0.30, 0.20, 0.40, 2],
    [0.15, 0.75, 0.10, 0.15, 0],
    [0.82, 0.60, 0.30, 0.25, 1],
], dtype=torch.float32)
    print(ground_truth.shape)
