import torch
from src.helpers import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, intersection_area


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

