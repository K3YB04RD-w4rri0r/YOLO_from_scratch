import torch

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

def intersection_area(b1 : torch.Tensor, b2 : torch.Tensor) -> torch.Tensor:
    xmin1, ymin1, xmax1, ymax1 = b1.unbind(-1)
    xmin2, ymin2, xmax2, ymax2 = b2.unbind(-1)
    return (torch.min(xmax1, xmax2) - torch.max(xmin1, xmin2)).clamp(min=0) * (torch.min(ymax1, ymax2) - torch.max(ymin1, ymin2)).clamp(min=0)

def IoU(b1, b2):
    xmin1, ymin1, xmax1, ymax1 = b1.unbind(-1)
    xmin2, ymin2, xmax2, ymax2 = b2.unbind(-1)

    A1 = (xmax1 - xmin1).clamp(min=0) * (ymax1 - ymin1).clamp(min=0)
    A2 = (xmax2 - xmin2).clamp(min=0) * (ymax2 - ymin2).clamp(min=0)
    I = intersection_area(b1, b2)

    return I / (A1 + A2 - I + 1e-6)

def test_box_conversion():
    xmin = torch.rand(100)
    ymin = torch.rand(100)

    xmax = xmin + torch.rand(100)
    ymax = ymin + torch.rand(100)

    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    assert torch.allclose(boxes,box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(boxes)))

if __name__ == "__main__":
    test_box_conversion()
    box2 = torch.tensor([0., 0., 10., 10.])
    box1 = torch.tensor([5., 5., 15., 15.])
    print(intersection_area(box1, box2))


    box3 = torch.tensor([0., 0., 10., 10.])
    box4 = torch.tensor([20., 20., 30., 30.])
    print(intersection_area(box3, box4))





