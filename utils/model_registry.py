# utils/model_registry.py

HF_TIMM_MODEL_MAP = {
    # ===== Clarity (blur / clear) =====
    "Mumulin/vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k-NTUH": {
        "timm_model": "vit_mediumd_patch16_reg4_gap_384",
        "img_size": 384,
        "num_classes": 2,
        "in_chans": 3,
        "checkpoint": "model_best.pth.tar",
    },

    # ===== NEED / NOT-NEED =====
    "Mumulin/beit3_large_patch16_224.in22k_ft_in1k-NTUH": {
        "timm_model": "beit3_large_patch16_224",
        "img_size": 224,
        "num_classes": 2,
        "in_chans": 3,
        "checkpoint": "model_best.pth.tar",
    },

    "Mumulin/beit3_base_patch16_224.indomain_pt-NTUH": {
        "timm_model": "beit3_base_patch16_224",
        "img_size": 224,
        "num_classes": 2,
        "in_chans": 3,
        "checkpoint": "model_best.pth.tar",
    },

    "Mumulin/swinv2_large_window12_192-NTUH": {
        "timm_model": "swinv2_large_window12_192",
        "img_size": 192,
        "num_classes": 2,
        "in_chans": 3,
        "checkpoint": "model_best.pth.tar",
    },
}
