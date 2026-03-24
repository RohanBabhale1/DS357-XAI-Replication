import torch
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient

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

data = torch.randn(1, 3, 224, 224)
model = vgg16_bn()
canonizers = [SequentialMergeBatchNorm()]
composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

with Gradient(model=model, composite=composite) as attributor:
    relevance_at_output = torch.eye(1000)[[0]]
    output, relevance = attributor(data, relevance_at_output)

print("OK — relevance shape:", relevance.shape)