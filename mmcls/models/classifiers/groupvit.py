import itertools
import os
import random

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, GroupViTModel

from .base import BaseClassifier
from .class_names import CLASS_NAME, preset_noun_prompt_templates, csp_templates, \
    preset_adj_prompt_templates, preset_noun_prompt_templates_for_sketch
from ..builder import CLASSIFIERS


@CLASSIFIERS.register_module()
class GroupViTClassifier(BaseClassifier):

    def __init__(self,
                 arch='ViT-B/16',
                 train_dataset=None,
                 wordnet_database=None,
                 txt_exclude=None,
                 neg_subsample=-1,
                 pos_neg_sim='neg_centroid',
                 neg_topk=10000,
                 emb_batchsize=1000,
                 init_cfg=None,
                 prompt_idx_pos=None,
                 prompt_idx_neg=None,
                 exclude_super_class=None,
                 dump_neg=False,
                 cls_mode=False,
                 ft_head=None,
                 load_dump_neg=False,
                 pretrained=None,
                 pencentile=1,
                 pos_topk=None,
                 seed=0):
        super(GroupViTClassifier, self).__init__(init_cfg)
        self.local_rank = os.environ['LOCAL_RANK']
        self.device = "cuda:{}".format(self.local_rank)

        self.model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        self.model.eval()

        self.cls_mode = cls_mode

        self.ft_head = None

        if prompt_idx_pos is None:
            prompt_idx_pos = -1
        if exclude_super_class is not None:
            class_name = CLASS_NAME[train_dataset][exclude_super_class]
        else:
            class_name = CLASS_NAME[train_dataset]
        if train_dataset == 'imagenet_sketch':
            noun_prompt_templates = preset_noun_prompt_templates_for_sketch
        else:
            noun_prompt_templates = preset_noun_prompt_templates
        self.id_prompts = [pair[0].format(pair[1]) for pair in list(itertools.product(noun_prompt_templates, class_name))]
        text_inputs_pos = self.tokenizer(self.id_prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.text_features_pos = self.model.get_text_features(**text_inputs_pos).to(torch.float32)
            self.feat_dim = self.text_features_pos.shape[-1]
            self.text_features_pos = self.text_features_pos.view(-1, len(class_name), self.feat_dim)
            self.text_features_pos = self.text_features_pos.mean(dim=0)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

        txtfiles = os.listdir(wordnet_database)
        if txt_exclude:
            file_names = txt_exclude.split(',')
            for file in file_names:
                txtfiles.remove(file)
        words_noun = []
        words_adj = []
        if prompt_idx_neg is None:
            prompt_idx_neg = -1
        prompt_templete = dict(
            adj=csp_templates,
            noun=noun_prompt_templates,
        )

        dedup = dict()
        print(f'Set Random Seed: {seed}')
        random.seed(seed)
        noun_length = 0
        adj_length = 0
        for file in txtfiles:
            filetype = file.split('.')[0]
            if filetype not in prompt_templete:
                continue
            with open(os.path.join(wordnet_database, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace('_', ' ')
                    if line.strip() in dedup:
                        continue
                    dedup[line.strip()] = None
                    if filetype == 'noun':
                        if pos_topk is not None:
                            if line.strip() in class_name:
                                continue
                        noun_length += 1
                        for template in prompt_templete[filetype]:
                            words_noun.append(template.format(line.strip()))
                    elif filetype == 'adj':
                        adj_length += 1
                        candidate = random.choice(prompt_templete[filetype]).format(line.strip())
                        for template in preset_adj_prompt_templates:
                            words_adj.append(template.format(candidate))
                    else:
                        raise TypeError

        words_total = words_noun + words_adj
        ensemble_noun_length = len(words_noun)

        print(f'Total noun number: {noun_length}')
        print(f'Total adj number: {adj_length}')


        with torch.no_grad():
            self.text_features_neg = []
            for i in tqdm(range(0, len(words_total), emb_batchsize)):
                text_inputs_neg_batch = self.tokenizer(words_total[i: i + emb_batchsize], padding=True, return_tensors="pt").to(self.device)
                x = self.model.get_text_features(**text_inputs_neg_batch).to(torch.float32)
                self.text_features_neg.append(x)
            self.text_features_neg = torch.cat(self.text_features_neg, dim=0)

            noun_text_features_neg = self.text_features_neg[:ensemble_noun_length].view(-1, len(noun_prompt_templates), self.feat_dim).mean(dim=1)
            adj_text_features_neg = self.text_features_neg[ensemble_noun_length:].view(-1, len(preset_adj_prompt_templates), self.feat_dim).mean(dim=1)
            self.text_features_neg = torch.cat([noun_text_features_neg, adj_text_features_neg], dim=0)

            self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

            neg_sim = []
            for i in range(0, noun_length + adj_length, emb_batchsize):
                tmp = self.text_features_neg[i: i + emb_batchsize] @ self.text_features_pos.T
                tmp = tmp.to(torch.float32)
                sim = torch.quantile(tmp, q=pencentile, dim=-1)
                maximum = torch.max(tmp, dim=1)[0]
                sim[maximum > 0.95] = 1.0
                neg_sim.append(sim)

            neg_sim = torch.cat(neg_sim, dim=0)
            neg_sim_noun = neg_sim[:noun_length]
            neg_sim_adj = neg_sim[noun_length:]
            text_features_neg_noun = self.text_features_neg[:noun_length]
            text_features_neg_adj = self.text_features_neg[noun_length:]

            ind_noun = torch.argsort(neg_sim_noun)
            ind_adj = torch.argsort(neg_sim_adj)

            text_features_neg_noun_selected = text_features_neg_noun[ind_noun[0:int(len(ind_noun) * neg_topk)]]
            text_features_neg_adj_selected = text_features_neg_adj[ind_adj[0:int(len(ind_adj) * neg_topk)]]
            self.text_features_neg = torch.cat([text_features_neg_noun_selected,text_features_neg_adj_selected], dim=0)

            self.words_noun_selected = [words_noun[i * len(noun_prompt_templates)] for i in ind_noun[0:int(len(ind_noun) * neg_topk)].tolist()]
            self.words_adj_selected = [words_adj[i * len(preset_adj_prompt_templates)] for i in ind_adj[0:int(len(ind_adj) * neg_topk)].tolist()]

            print(f'ID_length: {self.text_features_pos.shape[0]}')
            print(f'selected_noun_length: {text_features_neg_noun_selected.shape[0]}')
            print(f'selected_adj_length: {text_features_neg_adj_selected.shape[0]}')
            print(f'total_selected_neg_labels: {text_features_neg_noun_selected.shape[0] + text_features_neg_adj_selected.shape[0]}')

            self.adj_start_idx = int(len(ind_noun) * neg_topk)

    def extract_feat(self, img, stage='neck'):
        raise NotImplementedError

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True,
                    **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            # image_features = self.model.encode_image(img)
            image_features = self.model.get_image_features(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        if self.cls_mode:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            if self.ft_head is None:
                pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            else:
                pos_sim = self.ft_head(image_features)
            pos_sim = list(pos_sim.softmax(dim=-1).detach().cpu().numpy())
            return pos_sim
        else:
            image_features = image_features.to(torch.float32)
            self.text_features_pos = self.text_features_pos.to(torch.float32)
            self.text_features_neg = self.text_features_neg.to(torch.float32)
            if self.ft_head is None:
                pos_sim = (100.0 * image_features @ self.text_features_pos.T)
            else:
                pos_sim = self.ft_head(image_features)
            neg_sim = (100.0 * image_features @ self.text_features_neg.T)

            return pos_sim, neg_sim
