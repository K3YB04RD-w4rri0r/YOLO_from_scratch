import torch
import numpy as np

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)

    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(-1)

    cx = xmin + (xmax - xmin) / 2
    cy = ymin + (ymax - ymin) / 2

    w = (xmax - xmin)
    h = (ymax - ymin)

    return torch.stack([cx, cy, w , h], dim=-1)

def intersection_area(xmin1 : torch.Tensor, ymin1 : torch.Tensor, xmax1 : torch.Tensor, ymax1 : torch.Tensor,
                      xmin2 : torch.Tensor, ymin2 : torch.Tensor, xmax2 : torch.Tensor, ymax2 : torch.Tensor) -> torch.Tensor:
    # xmin1, ymin1, xmax1, ymax1 = b1.unbind(-1)
    # xmin2, ymin2, xmax2, ymax2 = b2.unbind(-1)
    return (torch.min(xmax1, xmax2) - torch.max(xmin1, xmin2)).clamp(min=0) * (torch.min(ymax1, ymax2) - torch.max(ymin1, ymin2)).clamp(min=0)

def IoU(b1 : torch.Tensor, b2 : torch.Tensor):
    xmin1 = b1[..., 0]
    ymin1 = b1[..., 1]
    xmax1 = b1[..., 2]
    ymax1 = b1[..., 3]

    xmin2 = b2[..., 0]
    ymin2 = b2[..., 1]
    xmax2 = b2[..., 2]
    ymax2 = b2[..., 3]

    A1 = (xmax1 - xmin1).clamp(min=0) * (ymax1 - ymin1).clamp(min=0)
    A2 = (xmax2 - xmin2).clamp(min=0) * (ymax2 - ymin2).clamp(min=0)
    I = intersection_area(xmin1, ymin1, xmax1, ymax1,
                          xmin2, ymin2, xmax2, ymax2)
    

    return I / (A1 + A2 - I + 1e-6)


def single_class_nms(boxes : torch.Tensor, scores : torch.Tensor, thresh : float) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    out = []
    while order.numel() > 0: # same as len since order (B,)
        best_box = boxes[order[0]]
        out.append(best_box)
        survivors_mask = IoU(best_box, boxes[order]) < thresh
        order = order[survivors_mask]

    return torch.stack(out)

       
def single_class_nms(boxes : torch.Tensor, scores : torch.Tensor, thresh : float) -> torch.Tensor:
    idxs = torch.argsort(scores, descending=True).tolist()
    keep = []

    while len(idxs) > 0:
        current = idxs.pop(0)
        keep.append(current)
        idxs = idxs[IoU(boxes[current], boxes[idxs]) < thresh]

    return torch.tensor(keep, dtype=torch.long)


def multiple_class_nms(boxes : torch.Tensor, classes : torch.Tensor, scores : torch.Tensor, thresh : float) -> torch.Tensor:
    # boxes -> (B, 4)
    # classes -> (B,)
    # scores -> (B,)
    unique_cls = torch.unique(classes)
    class_nms = []
    for c in unique_cls:
        mask = (classes == c)

        class_boxes     = boxes[mask]
        class_scores    = scores[mask]

        global_idx  = torch.nonzero(mask).squeeze(-1)
        class_elts  = single_class_nms(class_boxes, class_scores, thresh)
        class_nms.append(global_idx[class_elts])
    return torch.cat(class_nms)


def offset_mc_nms(boxes : torch.Tensor, classes : torch.Tensor, scores : torch.Tensor, thresh : float) -> torch.Tensor:
    pass

def test_box_conversion():
    xmin = torch.rand(100)
    ymin = torch.rand(100)
    xmax = xmin + torch.rand(100)
    ymax = ymin + torch.rand(100)
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    assert torch.allclose(boxes,box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(boxes)))

def test_intersection():
    box2 = torch.tensor([0., 0., 10., 10.])
    box1 = torch.tensor([5., 5., 15., 15.])
    print(intersection_area(box1, box2))


    box3 = torch.tensor([0., 0., 10., 10.])
    box4 = torch.tensor([20., 20., 30., 30.])
    print(intersection_area(box3, box4))





if __name__ == "__main__":
    # test_box_conversion()
    xmin = torch.rand(100)
    ymin = torch.rand(100)
    xmax = xmin + torch.rand(100)
    ymax = ymin + torch.rand(100)
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    scores = torch.tensor([torch.randn(size=(1,)) for _ in range(100)])
    single_class_nms(boxes=boxes, scores=scores, thresh=0.5)





