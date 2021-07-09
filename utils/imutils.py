

def mask_overlay(pred, image, alpha=0.3):
    return alpha * image + (1 - alpha) * pred