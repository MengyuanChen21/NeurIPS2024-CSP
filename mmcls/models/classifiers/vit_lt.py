import os

from timm.models import create_model, load_checkpoint

from .base import BaseClassifier
from ..builder import CLASSIFIERS


@CLASSIFIERS.register_module()
class VitClassifier(BaseClassifier):

    def __init__(self,
                 model='ViT-B/16',
                 checkpoint=None,
                 num_classes=None,
                 gp=None,
                 init_cfg=None):
        super(VitClassifier, self).__init__(init_cfg)
        self.local_rank = os.environ['LOCAL_RANK']
        self.device = "cuda:{}".format(self.local_rank)
        self.model = create_model(
            model,
            pretrained=False,
            num_classes=num_classes,
            in_chans=3,
            global_pool=gp,
            scriptable=False,
        )
        load_checkpoint(self.model, checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()


    def extract_feat(self, img, stage='neck'):
        # TODO
        raise NotImplementedError

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True, **kwargs):
        """Test without augmentation."""
        # with torch.no_grad():
        output = self.model(img)

        output = output[:,:1000]
        return output

