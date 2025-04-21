# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List

import numpy as np
from mmengine.fileio import get_local_path
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class SingleCategoryRefDataset(BaseDetDataset):
    """Dataset for *grouped* referring‑expression JSON produced by
    `convert_to_grouped_odvg`.  It expects ``ann_file`` to contain  ::

        {
          "images": [
             { "file_name": ..., "height": ..., "width": ...,
               "caption": "...",
               "category_id": int,
               "boxes": [ [x,y,w,h], ... ] }
             ...
          ],
          "categories": [...]
        }

    Each *image entry* can contain multiple GT boxes (`boxes` list), all sharing
    the same caption and category.
    """

    META_KEYS = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                 'scale_factor', 'text', 'custom_entities',
                 'tokens_positive')

    def load_data_list(self) -> List[dict]:
        # read grouped JSON
        with get_local_path(self.ann_file,
                            backend_args=self.backend_args) as pth:
            j = Path(pth).read_text()
        import json
        root = json.loads(j)

        data_infos = []
        for img in root['images']:
            data_info = dict()

            img_path = Path(self.data_prefix['img']) / img['file_name']
            data_info['img_path'] = str(img_path)
            data_info['img_id'] = img['id']
            data_info['height'] = img['height']
            data_info['width'] = img['width']
            data_info['dataset_mode'] = img.get('dataset_name', 'grouped')

            # referring text
            data_info['text'] = img['caption']
            data_info['custom_entities'] = False
            data_info['tokens_positive'] = -1

            # build instance list from ``boxes``
            boxes = np.asarray(img['boxes'], dtype=float)
            cat_id = int(img['category_id'])
            instances = []
            for box in boxes:
                x, y, w, h = box.tolist()
                instances.append(dict(
                    bbox=[x, y, x + w, y + h],
                    bbox_label=cat_id,
                    ignore_flag=0,
                ))
            data_info['instances'] = instances
            data_infos.append(data_info)

        return data_infos