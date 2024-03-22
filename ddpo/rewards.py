# Following closure style for rewards functions using in https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/rewards.py

def aesthetic_score():
    from ddpo.laion_aesthetic import AestheticRewardModel

    laion_aesthetic = AestheticRewardModel(model_checkpoint="ViT-L/14",
                                           device="cuda")
    
    def _fn(images):
        aesthetic_score = laion_aesthetic(images)
        return aesthetic_score
    
    return _fn
