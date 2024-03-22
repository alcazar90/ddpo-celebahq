from ddpo.utils import decode_tensor_to_np_img

# Following closure style for rewards functions using in https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/rewards.py

def aesthetic_score():
    from ddpo.laion_aesthetic import AestheticRewardModel

    laion_aesthetic = AestheticRewardModel(model_checkpoint="ViT-L/14",
                                           device="cuda")
    
    def _fn(images):
        aesthetic_score = laion_aesthetic(images)
        return aesthetic_score
    
    return _fn


def under30_old():
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    # Obtain id2label and label2id mapping
    # model.config.id2label, model.config.label2id

    def _fn(images):
        inputs = transforms(
            decode_tensor_to_np_img(images,
                                    melt_batch=False,), 
            return_tensors="pt")
        outputs = model(**inputs).logits    
        probs = outputs.softmax(dim=1)
        rewards = (probs.argmax(dim=1) <= 3).float()
        return rewards

    return _fn


def over50_old():
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
    transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    # obtain id2label and label2id mapping
    # model.config.id2label, model.config.label2id

    def _fn(images):
        inputs = transforms(
            decode_tensor_to_np_img(images,
                                    melt_batch=False,), 
            return_tensors="pt")
        outputs = model(**inputs).logits    
        probs = outputs.softmax(dim=1)
        rewards = (probs.argmax(dim=1) >= 6).float()
        return rewards

    return _fn
    

    
