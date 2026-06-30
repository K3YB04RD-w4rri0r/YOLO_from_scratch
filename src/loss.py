import torch

def continuous_box_center(cx: torch.Tensor, cy: torch.Tensor, S: int):
    "assuming normalized inputs relative to image size"
    cell_x = (cx * S).floor().long()
    cell_y = (cy * S).floor().long()
    return cell_x, cell_y


def cell_offset(cx: torch.Tensor, cy: torch.Tensor, S):
    "assuming normalized inputs relative to image size"
    cx_s = cx * S
    cy_s = cy * S
    return cx_s - cx_s.floor(), cy_s - cy_s.floor()

def ground_truth_transform(gt_boxes: torch.Tensor, S : int, C, B, BATCH):
    "normalized inputs, boxes as (image_id_in_minibatch, x_center, y_center, w, h, class_id) -> (image_id, i, j, B * (boxes - class + confidence )flattened + one hot of class)"
    transformation = torch.zeros(size = (BATCH, S, S, B * 5 + C))
    batch_idx , cx, cy , w, h, class_id = gt_boxes.unbind(-1)
    batch_idx = batch_idx.long()
    class_id = class_id.long()  
    
    j, i = continuous_box_center(cx, cy, S)
    cell_x, cell_y = cell_offset(cx, cy, S)
    confidence = torch.ones_like(cell_x)

    transformation[batch_idx, j, i, 0: 5] =  torch.stack([cell_x, cell_y, w, h, confidence], dim=-1).to(transformation.device) # note: silent overwriting, no logic for keeping different gt boxes
    transformation[batch_idx, j, i, B * 5 + class_id] = 1.0

    return transformation

def select_true_boxes(target_tensor : torch.Tensor) -> torch.Tensor:
    contains_object = (target_tensor[..., 4] > 0)
    final_boxes = target_tensor[contains_object]

    return final_boxes






    






if __name__ == "__main__":
    BATCH, S, B, C = 32, 7, 2, 20
    prediction = torch.randn(size=(BATCH, S,S, (B*5 + C)))
    # (batch_idx, x_center, y_center, w, h, class_id)
    gt_boxes = torch.tensor([
    [0, 0.42, 0.31, 0.20, 0.10, 3],
    [0, 0.80, 0.60, 0.15, 0.25, 7],])
    

    predictions = torch.rand(BATCH, S, S, B * 5 + C)