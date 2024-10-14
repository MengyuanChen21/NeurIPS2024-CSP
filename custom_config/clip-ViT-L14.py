method_name = 'ScalableClassifier'
model_name = 'ViT-L/14'

train_dataset = 'Balance'
custom_name = "Official"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name).replace('/','_')
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset).replace('/','_')
quick_test = False
# quick_test = True

model = dict(
    type=method_name,
    debug_mode=False,
    t=1,
    ngroup=100,
    group_fuse_num=None,
    classifier=dict(
        type='CLIPScalableClassifier',
        arch=model_name,
        train_dataset='imagenet',
        wordnet_database='/data_SSD1/cmy/csp-submit/txtfiles/',
        txt_exclude=
        'noun.person.txt,noun.quantity.txt,noun.group.txt,'
        'adj.pert.txt',
        neg_subsample=-1,
        pos_neg_sim='neg_centroid',
        neg_topk=0.15,  # percentage
        emb_batchsize=1000,
        prompt_idx_pos=85,  # [0,80]
        prompt_idx_neg=85,  # [0,80]
        # dump_neg=True,
        dump_neg=False,
        # load_dump_neg=True,
        load_dump_neg=False,
        pencentile=0.95
    )
)
pipline =[
          dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        # path='/data/val',
        path='/data_SSD2/cmy/ImageNet/ILSVRC/Data/CLS-LOC/val',
        # data_ann='/data/meta/val_labeled.txt',
        data_ann='/data_SSD2/cmy/ImageNet/ILSVRC/val_annotation_prefix.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        train_label=None,
    ),
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            # path='/data/ood_data/iNaturalist/images',
            path='/data_SSD2/cmy/iNaturalist/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            # path='/data/ood_data/SUN/images',
            path='/data_SSD2/cmy/SUN/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            # path='/data/ood_data/Places/images',
            path='/data_SSD2/cmy/Places/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            # path='/data/ood_data/Textures/dtd/images_collate',
            path='/data_SSD2/cmy/Textures/images_collate',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
work_dir = './results/'
