# mmdet/evaluation/metrics/single_category_ref_metric.py
from typing import Dict, Optional, Sequence, Any

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from ..functional import bbox_overlaps


@METRICS.register_module()
class SingleCategoryRefMetric(BaseMetric):
    """Compute average confidence *gap* between the correct box and
    same‑category distractor boxes for single‑category ref‑exp benchmarks."""

    default_prefix: Optional[str] = 'single_cat_ref'

    def __init__(self, iou_thrs: float = 0.5, **kwargs: Any) -> None:
        # Strip keys that BaseMetric does not accept
        for k in ('metric', 'format_only', 'backend_args', 'ann_file'):
            kwargs.pop(k, None)
        super().__init__(**kwargs)
        self.iou_thrs = iou_thrs

    # ------------------------------------------------------------------
    # Collect per‑image results
    # ------------------------------------------------------------------
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            preds = sample['pred_instances']

            # Ground‑truth boxes from LoadAnnotations
            gt_inst = sample.get('gt_instances', None)
            if gt_inst is None:
                continue

            if hasattr(gt_inst, 'bboxes'):
                all_boxes = gt_inst.bboxes.cpu().numpy()
            else:  # plain dict
                tensor_boxes = gt_inst['bboxes']
            if torch.is_tensor(tensor_boxes):
                tensor_boxes = tensor_boxes.cpu()
            all_boxes = np.asarray(tensor_boxes, dtype=float)

            if all_boxes.size == 0:
                continue

            gt_bbox = all_boxes[0:1]          # (1,4)
            distractor_bboxes = all_boxes[1:]  # (K,4)

            self.results.append({
                'gt_bbox': gt_bbox,
                'distractor_bboxes': distractor_bboxes,
                'bboxes': preds['bboxes'].cpu().numpy(),
                'scores': preds['scores'].cpu().numpy(),
            })

    # ------------------------------------------------------------------
    # Final aggregation
    # ------------------------------------------------------------------
    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if len(results) == 0:
            logger.warning('SingleCategoryRefMetric got empty results list.')
            return {'avg_gap': 0.0, 'frac_positive_gap': 0.0}

        gaps = []
        for r in results:
            bboxes, scores = r['bboxes'], r['scores']

            iou_t = bbox_overlaps(bboxes, r['gt_bbox'])[:, 0]
            iou_d = bbox_overlaps(bboxes, r['distractor_bboxes']) if r['distractor_bboxes'].size else np.zeros((bboxes.shape[0], 0))

            s_t = scores[iou_t >= self.iou_thrs].max(initial=0.0)
            mask_d = (iou_d >= self.iou_thrs).any(axis=1) if iou_d.size else np.zeros(bboxes.shape[0], dtype=bool)
            s_d = scores[mask_d].max(initial=0.0)
            gaps.append(s_t - s_d)

        gaps = np.array(gaps, dtype=float)
        metrics = {
            'avg_gap': float(gaps.mean()),
            'frac_positive_gap': float((gaps > 0).mean()),
        }
        logger.info(metrics)
        return metrics
