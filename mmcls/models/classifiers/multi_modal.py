import itertools
import os
import random

import clip
import torch
from tqdm import tqdm

from .base import BaseClassifier
from .class_names import CLASS_NAME, preset_noun_prompt_templates, csp_templates, \
    preset_adj_prompt_templates, preset_noun_prompt_templates_for_sketch
from ..builder import CLASSIFIERS


@CLASSIFIERS.register_module()
class CLIPScalableClassifier(BaseClassifier):
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
        super(CLIPScalableClassifier, self).__init__(init_cfg)
        self.local_rank = os.environ['LOCAL_RANK']
        self.device = "cuda:{}".format(self.local_rank)

        self.clip_model, _ = clip.load(arch, self.device, jit=False)
        self.clip_model.eval()
        self.model = self.clip_model
        self.cls_mode = cls_mode

        self.ft_head = None
        print(f'Select {int(neg_topk * 100)}% candidates.')

        if exclude_super_class is not None:
            class_name = CLASS_NAME[train_dataset][exclude_super_class]
        else:
            class_name = CLASS_NAME[train_dataset]
        if train_dataset == 'imagenet_sketch':
            noun_prompt_templates = preset_noun_prompt_templates_for_sketch
        else:
            noun_prompt_templates = preset_noun_prompt_templates
        self.id_prompts = [pair[0].format(pair[1]) for pair in list(itertools.product(noun_prompt_templates, class_name))]
        text_inputs_pos = torch.cat([clip.tokenize(f"{c}") for c in self.id_prompts]).to(self.device)
        with torch.no_grad():
            if arch == 'RN50x64':  # Save GPU space
                text_features_pos_part1 = self.clip_model.encode_text(text_inputs_pos[:int(text_inputs_pos.shape[0] / 2)]).to(torch.float32)
                text_features_pos_part2 = self.clip_model.encode_text(text_inputs_pos[int(text_inputs_pos.shape[0] / 2):]).to(torch.float32)
                self.text_features_pos = torch.cat([text_features_pos_part1, text_features_pos_part2], dim=0)
            else:
                self.text_features_pos = self.clip_model.encode_text(text_inputs_pos).to(torch.float32)
            self.feat_dim = self.text_features_pos.shape[-1]
            self.text_features_pos = self.text_features_pos.view(-1, len(class_name), self.feat_dim)
            self.text_features_pos = self.text_features_pos.mean(dim=0)
            self.text_features_pos /= self.text_features_pos.norm(dim=-1, keepdim=True)

        embedding_save_path = './neg_embedding'
        if not load_dump_neg or not os.path.exists(f'{embedding_save_path}/neg_dump.pth'):
            txtfiles = os.listdir(wordnet_database)
            if txt_exclude:
                file_names = txt_exclude.split(',')
                for file in file_names:
                    txtfiles.remove(file)
            words_noun = []
            words_adj = []
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

            if neg_subsample > 0:
                random.seed(42)
                words_noun = random.sample(words_noun, neg_subsample)

            text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in tqdm(words_noun)]).to(self.device)
            text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in tqdm(words_adj)]).to(self.device)
            text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
            ensemble_noun_length = len(text_inputs_neg_noun)

            print(f'Candidate pool size: {noun_length + adj_length}')
            print(f'Noun candidates: {noun_length}')
            print(f'Adj candidates: {adj_length}')

            with torch.no_grad():
                self.text_features_neg = []
                for i in tqdm(range(0, len(text_inputs_neg), emb_batchsize)):
                    x = self.clip_model.encode_text(text_inputs_neg[i: i + emb_batchsize])
                    self.text_features_neg.append(x)
                self.text_features_neg = torch.cat(self.text_features_neg, dim=0)

                noun_text_features_neg = self.text_features_neg[:ensemble_noun_length].view(-1, len(noun_prompt_templates), self.feat_dim).mean(dim=1)
                adj_text_features_neg = self.text_features_neg[ensemble_noun_length:].view(-1, len(preset_adj_prompt_templates), self.feat_dim).mean(dim=1)
                self.text_features_neg = torch.cat([noun_text_features_neg, adj_text_features_neg], dim=0)

                self.text_features_neg /= self.text_features_neg.norm(dim=-1, keepdim=True)

                if dump_neg:
                    tmp = self.text_features_neg.cpu()
                    dump_dict = dict(neg_emb=tmp, noun_length=noun_length, adj_length=adj_length)
                    os.makedirs(embedding_save_path, exist_ok=True)
                    torch.save(dump_dict, f'{embedding_save_path}/neg_dump.pth')
                    print(f'Save the negative embedding as {embedding_save_path}/neg_dump.pth')
        else:
            # If you saved the negative embedding file with dump_neg=True,
            # you can directly load it to save time the next time you run the code.
            dump_dict = torch.load(f'{embedding_save_path}/neg_dump.pth')
            self.text_features_neg = dump_dict['neg_emb'].to(self.device)
            noun_length = dump_dict['noun_length']
            adj_length = dump_dict['adj_length']

        with torch.no_grad():
            self.text_features_neg = self.text_features_neg.to(torch.float32)

            if pos_neg_sim == 'neg_centroid':
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

                print(f'ID_length: {self.text_features_pos.shape[0]}')
                print(f'selected_noun_length: {text_features_neg_noun_selected.shape[0]}')
                print(f'selected_adj_length: {text_features_neg_adj_selected.shape[0]}')
                print(f'total_selected_neg_labels: {text_features_neg_noun_selected.shape[0] + text_features_neg_adj_selected.shape[0]}')

            self.adj_start_idx = int(len(ind_noun) * neg_topk)

            self.pos_score = 0
            self.noun_score = 0
            self.adj_score = 0
            self.sample_num = 0
            self.last_batch_dataset = None

    def extract_feat(self, img, stage='neck'):
        raise NotImplementedError

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas=None, require_features=False, require_backbone_features=False, softmax=True,
                    **kwargs):
        """Test without augmentation."""
        with torch.no_grad():
            image_features = self.model.encode_image(img)
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


def split_path(path):
    """Split a file path into all of its parts."""
    parts = []
    while path:
        path, tail = os.path.split(path)
        if tail:
            parts.append(tail)
        else:  # If there's no more tail, the head is the last part (root)
            if path:
                parts.append(path)
            break
    return parts[::-1]  # Reverse the list to get the correct order
