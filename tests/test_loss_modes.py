"""Numerical checks for loss selection, corner normalization and preserved DFL."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1]/'.runtime'))
import torch
from ultralytics.utils.loss import BboxLoss, bbox_mpdiou
from ultralytics.utils.metrics import bbox_iou


class LossModes(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'USE_NWD':'False','BBOX_LOSS_TYPE':'ciou'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_mpdiou_analytic_and_scale_invariance(self):
        a=torch.tensor([[0.,0.,2.,2.]])
        b=torch.tensor([[1.,0.,3.,2.]])
        expected=torch.tensor([[1/3-2/200]])
        torch.testing.assert_close(bbox_mpdiou(a,b,(10,10)),expected)
        torch.testing.assert_close(bbox_mpdiou(a*8,b*8,(80,80)),expected)
        torch.testing.assert_close(bbox_mpdiou(a,a,(10,10)),torch.ones(1,1))
        far=torch.tensor([[5.,5.,7.,7.]],requires_grad=True)
        value=bbox_mpdiou(far,a,(10,10))
        self.assertLess(value.item(),0)
        value.sum().backward()
        self.assertTrue(torch.isfinite(far.grad).all())
        self.assertGreater(far.grad.abs().sum().item(),0)

    def test_modes_preserve_dfl_and_use_pixel_units(self):
        torch.manual_seed(42)
        pred=torch.tensor([[[0.,0.,2.,2.],[0.,0.,2.,2.]]],requires_grad=True)
        target=pred.detach()+.25
        scores=torch.ones(1,2,1)
        mask=torch.ones(1,2,dtype=torch.bool)
        strides=torch.tensor([[8.],[32.]])
        dist=torch.randn(1,2,64,requires_grad=True)
        args=(dist,pred,torch.ones(2,2),target,scores,scores.sum(),mask)
        ciou=BboxLoss(16,box_loss_type='ciou')
        mpd=BboxLoss(16,box_loss_type='mpdiou')
        nwd=BboxLoss(16,use_nwd=True,nwd_weight=.5,box_loss_type='ciou')
        old_box,old_dfl=ciou(*args,stride_tensor=strides,image_size=(640,640))
        expected_ciou=(1-bbox_iou(pred[mask],target[mask],xywh=False,CIoU=True)).mean()
        torch.testing.assert_close(old_box,expected_ciou)
        mp_box,mp_dfl=mpd(*args,stride_tensor=strides,image_size=(640,640))
        expected_mp=(1-bbox_mpdiou(pred[mask]*strides,target[mask]*strides,(640,640))).mean()
        torch.testing.assert_close(mp_box,expected_mp)
        _,nwd_dfl=nwd(*args,stride_tensor=strides,image_size=(640,640))
        torch.testing.assert_close(mp_dfl,old_dfl,rtol=0,atol=0)
        torch.testing.assert_close(nwd_dfl,old_dfl,rtol=0,atol=0)
        (mp_box+mp_dfl).backward()
        self.assertTrue(torch.isfinite(pred.grad).all() and torch.isfinite(dist.grad).all())

    def test_invalid_modes_fail(self):
        with self.assertRaises(ValueError): BboxLoss(box_loss_type='wrong')
        with self.assertRaises(ValueError): BboxLoss(use_nwd=True,box_loss_type='mpdiou')
        with self.assertRaises(ValueError): bbox_mpdiou(torch.ones(1,4),torch.ones(1,4),(0,640))


if __name__=='__main__':unittest.main(verbosity=2)
