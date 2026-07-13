from .timeline_contrastive import Model as TimelineContrastive
from .timeline_ce_dot import Model as TimelineCEDot
from .ce_cls import Model as CECls
from .regression import Model as Regression

LOSSES = {
    "regression": Regression,
    "ce_year": CECls,
    "ce_century": CECls,
    "timeline_ce_dot": TimelineCEDot,
    "timeline_contrastive": TimelineContrastive,
}
