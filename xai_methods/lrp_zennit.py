import torch
import torch.nn.functional as F
from zennit.composites import EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1
from zennit.canonizers import SequentialMergeBatchNorm, MergeBatchNorm
from zennit.attribution import Gradient
from zennit.image import imgify

# Fix: patch F.batch_norm to floor eps at 1e-10
# (Zennit's canonizer sets eps=0.0 after merging BN, which PyTorch 2.x rejects)
_original_batch_norm = F.batch_norm
def _patched_batch_norm(input, running_mean, running_var, weight=None,
                        bias=None, training=False, momentum=0.1, eps=1e-5):
    if eps == 0.0:
        eps = 1e-10
    return _original_batch_norm(input, running_mean, running_var,
                                weight, bias, training, momentum, eps)
F.batch_norm = _patched_batch_norm


def get_composite(composite_name, model_name='vgg16_bn'):
    """
    Get Zennit composite rule for the given model.
    Uses SequentialMergeBatchNorm for VGG, MergeBatchNorm for ResNet.
    """
    if 'resnet' in model_name.lower():
        canonizers = [MergeBatchNorm()]
    else:
        canonizers = [SequentialMergeBatchNorm()]

    composites = {
        'EpsilonGammaBox':    EpsilonGammaBox(low=-3., high=3., canonizers=canonizers),
        'EpsilonPlus':        EpsilonPlus(canonizers=canonizers),
        'EpsilonAlpha2Beta1': EpsilonAlpha2Beta1(canonizers=canonizers),
    }
    return composites.get(composite_name)


def compute_lrp(model, image_tensor, composite_name, model_name='vgg16_bn', target_class=None):
    """
    Compute LRP attribution using Zennit (paper Listing 1).

    Args:
        model: pretrained PyTorch model in eval mode
        image_tensor: input tensor of shape (1, 3, 224, 224), normalized
        composite_name: one of 'EpsilonGammaBox', 'EpsilonPlus', 'EpsilonAlpha2Beta1'
        model_name: used to select correct canonizer
        target_class: ImageNet class index. If None, uses predicted class.

    Returns:
        relevance tensor of shape (1, 3, 224, 224)
    """
    model.eval()

    if target_class is None:
        with torch.no_grad():
            target_class = model(image_tensor).argmax().item()

    composite = get_composite(composite_name, model_name)

    if composite is not None:
        with Gradient(model=model, composite=composite) as attributor:
            relevance_at_output = torch.eye(1000)[[target_class]]
            output, relevance = attributor(image_tensor, relevance_at_output)
    else:
        # Plain gradient fallback
        image_tensor = image_tensor.clone().requires_grad_(True)
        output = model(image_tensor)
        relevance_at_output = torch.eye(1000)[[target_class]]
        (output * relevance_at_output).sum().backward()
        relevance = image_tensor.grad.detach()

    return relevance


def heatmap_to_image(relevance):
    """
    Convert relevance tensor to PIL image using Zennit's coldnhot colormap.
    Matches the paper's Figure 5/6 visual style:
      - Red/yellow = positive relevance (important regions)
      - Black = zero relevance
      - Blue = negative relevance
    """
    return imgify(relevance.sum(1), symmetric=True, cmap='coldnhot')


if __name__ == '__main__':
    print("lrp_zennit module loaded OK")