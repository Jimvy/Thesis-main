import torch
from torch import Tensor
import torch.nn.functional as F

# TODO don't forget to update me!!!
__all__ = [
    'ce_loss_acts', 'ce_loss_probs', 'ce_loss',
    'ce_loss_grad', 'ce_loss_grad_acts', 'ce_loss_grad_probs',
    'hkd_loss_acts', 'hkd_loss_probs', 'hkd_loss',
    'hkd_loss_grad', 'hkd_loss_grad_as', 'hkd_loss_grad_ps', 'hkd_loss_grad_temp'
]


def softmaxify(acts, t=None):
    if t is None:
        return F.softmax(acts, dim=1)
    else:
        return F.softmax(acts/t, dim=1)


def logsoftmaxify(acts, t=None):
    if t is None:
        return F.log_softmax(acts, dim=1)
    else:
        return F.log_softmax(acts/t, dim=1)


def opt_softmaxify(ins: Tensor, in_type='acts', t=None):
    if in_type == 'acts':
        return softmaxify(ins, t)
    elif in_type == 'probs':
        return ins.clone()


def opt_logsoftmaxify(ins: Tensor, in_type='acts', t=None):
    if in_type == 'acts':
        return logsoftmaxify(acts=ins, t=t)
    elif in_type == 'probs':
        return torch.log(ins)  # Numerically unstable...
    else:
        return ins.clone()


def ce_loss(ins: Tensor, in_type='acts'):
    r"""
    Computes standard cross-entropy loss, with target=0
    :param ins: activations of the student, shape BxC, either logits/activations,
                   or direct class "probabilities"
    :param in_type: type of the input: 'acts' or 'logits' if activations, 'probs' if class probabilities.
    :return: a tensor of shape Bx1 representing the loss
    """
    if in_type in ('acts', 'logits'):
        return ce_loss_acts(acts=ins)
    else:
        return ce_loss_probs(probs=ins)


def ce_loss_acts(acts: Tensor):
    r"""
    Computes standard cross-entropy loss, with target=0
    :param acts: the activation of the student, BxC logits
    :return: a tensor with B values representing the loss
    """
    logprobs = logsoftmaxify(acts)
    return _ce_loss_logprobs(logprobs)


def ce_loss_probs(probs: Tensor):
    r"""
    Computes standard cross-entropy loss, with target=0
    :param probs: the class "probabilities" of the student, BxC values between 0 and 1
    :return: a tensor with B values representing the loss
    """
    return _ce_loss_logprobs(torch.log(probs))


def _ce_loss_logprobs(logprobs: Tensor, target_class=0):
    r"""
    Computes standard cross-entropy loss, with target=0
    :param logprobs: the class log probas of the student, BxC values between -infinity and 0
    :return: a tensor with B values representing the loss
    """
    b = logprobs[:, target_class].unsqueeze(dim=1)
    ret = (-1) * b
    return ret


def ce_loss_grad(ins: Tensor, in_type='acts', target_class=0):
    r"""
    Computes gradient of CE loss wrt network input, respecting the type.
    That is, if the input are activations, computes the gradient w.r.t. student activations.
    :param ins: network input, BxC tensor, either activations or probs.
    :param in_type: type of the input: 'acts' if activations, 'probs' if class "probabilities" (output of softmax)
    :param target_class: the target class for these samples. Default 0
    :return: gradients, Tensor of BxC values
    """
    if in_type == 'acts':
        return ce_loss_grad_acts(ins, in_type, target_class)
    else:
        return ce_loss_grad_probs(ins, in_type, target_class)


def ce_loss_grad_acts(s_ins: Tensor, s_type='acts', target_class=0):
    r"""
    Computes gradient of CE loss wrt activations.
    Makes sense only if the input itself are logits.
    :param s_ins: activations of the student, BxC logits
    :param s_type: type of the input: 'acts' if activations, 'probs' if class "probabilities" (output of softmax)
    :param target_class: the target class for these samples. Default 0
    :return: gradients, Tensor of BxC values
    """
    s_probs = opt_softmaxify(s_ins, in_type=s_type)
    s_probs[:, target_class] -= 1
    return s_probs


def ce_loss_grad_probs(ins: Tensor, type='probs', target_class=0):
    r"""
    Computes gradient of CE loss wrt class probabilities,
    Makes sense only if the input itself are class probabilities.
    :param ins: activations of the student, BxC logits
    :param type: type of the input: 'acts' if activations, 'probs' if class "probabilities" (output of softmax)
    :param target_class: the target class for these samples. Default 0
    :return: gradients, Tensor of BxC values
    """
    as_probs = opt_softmaxify(ins, in_type=type)
    ret = (1 / as_probs[:, target_class]).reshape(as_probs.shape[0], 1)
    ret = ret.repeat(1, as_probs.shape[1])
    ret[:, target_class] *= (-1)
    return ret


def hkd_loss(s_ins: Tensor, t_ins: Tensor, t, s_in_type='acts', t_in_type='acts', tau_pow=2):
    r"""
    Computes Hinton knowledge distillation loss from activations/logits
    :param s_ins: B x C logits/probs from the student
    :param t_ins: B x C logits/probs from the teacher
    :param t: temperature, B or 1 value.
    :param s_in_type: type of the input, 'acts' or 'logits' if activations, 'probs' if class probabilities.
                    This type should be common to both tensors.
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: a tensor with B values representing the loss
    """
    """if s_in_type == t_in_type:
        if s_in_type in ('acts', 'logits'):
            return hkd_loss_acts(s_acts=s_ins, t_acts=t_ins, t=t, tau_pow=tau_pow)
        else:
            return hkd_loss_probs(s_probs=s_ins, t_probs=t_ins, t=t, tau_pow=tau_pow)"""
    s_logprobs = opt_logsoftmaxify(ins=s_ins, in_type=s_in_type, t=t)
    t_probs = opt_softmaxify(ins=t_ins, in_type=t_in_type, t=t)
    return _hkd_loss_logprobs(s_logprobs=s_logprobs, t_probs=t_probs, t=t, tau_pow=tau_pow)


def hkd_loss_acts(s_acts: Tensor, t_acts: Tensor, t, tau_pow=2):
    r"""
    Computes Hinton knowledge distillation loss from activations/logits
    :param s_acts: B x C logits from the student
    :param t_acts: B x C logits from the teacher
    :param t: temperature, B or 1 value.
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: a tensor with B values representing the loss
    """
    t_probs = softmaxify(t_acts, t=t)
    s_logprobs = logsoftmaxify(s_acts, t=t)
    return _hkd_loss_logprobs(s_logprobs=s_logprobs, t_probs=t_probs, t=t, tau_pow=tau_pow)


def hkd_loss_probs(s_probs: Tensor, t_probs: Tensor, t, tau_pow=2):
    r"""
    Computes Hinton knowledge distillation loss from class probabilities
    :param s_probs: B x C class probas from the student
    :param t_probs: B x C class probas from the teacher
    :param t: temperature, B or 1 value.
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: a tensor with B values representing the loss
    """
    return _hkd_loss_logprobs(t_probs=t_probs, s_logprobs=torch.log(s_probs), t=t, tau_pow=tau_pow)


def _hkd_loss_logprobs(s_logprobs: Tensor, t_probs: Tensor, t, tau_pow=2):
    r"""
    Computes Hinton knowledge distillation loss from class probabilities
    :param s_logprobs: B x C class log probas from the student
    :param t_probs: B x C class probas from the teacher
    :param t: temperature, B or 1 value.
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: a tensor with B values representing the loss
    """
    c = (t_probs * s_logprobs)
    d = c.sum(dim=1).unsqueeze(dim=1)
    ret = (-1) * (t**tau_pow) * d
    return ret


def hkd_loss_grad(s_ins: Tensor, t_ins: Tensor, t=1, s_in_type='acts', t_in_type='acts', tau_pow=2):
    r"""
    Computes the gradient of the HKD loss w.r.t. the student inputs, respecting the type of it.
    That is, if we are passed in the student activations, will return the gradient of the loss
    w.r.t. the student activations.
    :param s_ins: BxC logits/probs from the student
    :param t_ins: BxC logits/probs from the teacher
    :param t: temperature of knowledge distillation.
    :param s_in_type: type of the student input, 'acts' for activations/logits, 'probs' for class probabilities
    :param t_in_type: type of the teacher input, 'acts' for activations/logits, 'probs' for class probabilities
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: BxC tensor, with the gradient of the loss w.r.t. each component of the student input.
    """
    if s_in_type == 'acts':
        return hkd_loss_grad_as(s_ins, t_ins, t, s_in_type, t_in_type, tau_pow=tau_pow)
    else:
        return hkd_loss_grad_ps(s_ins, t_ins, t, s_in_type, t_in_type, tau_pow=tau_pow)


def hkd_loss_grad_as(s_ins: Tensor, t_ins: Tensor, t, s_in_type='acts', t_in_type='acts', tau_pow=2):
    r"""
    Computes the gradient of the HKD loss w.r.t. the student activations.
    :param s_ins: BxC logits/probs from the student
    :param t_ins: BxC logits/probs from the teacher
    :param t: temperature of knowledge distillation.
    :param s_in_type: type of the student input, 'acts' for activations/logits, 'probs' for class probabilities
    :param t_in_type: type of the teacher input, 'acts' for activations/logits, 'probs' for class probabilities
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: BxC tensor, with the gradient of the loss w.r.t. each component of the student activation.
    """
    s_probs = opt_softmaxify(s_ins, in_type=s_in_type, t=t)
    t_probs = opt_softmaxify(t_ins, in_type=t_in_type, t=t)
    return (t**(tau_pow-1)) * (s_probs - t_probs)


def hkd_loss_grad_ps(s_ins: Tensor, t_ins: Tensor, t, s_in_type='probs', t_in_type='probs', tau_pow=2):
    r"""
    Computes the gradient of the HKD loss w.r.t. the student class "probabilities".
    This gradient is zero where the loss is minimal, and was computed knowing the relation p_S[0] + p_S[1] = 1.
    For now, this gradient requires C=2, as the case C!=2 is not well defined.
    :param s_ins: BxC logits/probs from the student
    :param t_ins: BxC logits/probs from the teacher
    :param t: temperature of knowledge distillation.
    :param s_in_type: type of the student input, 'acts' for activations/logits, 'probs' for class probabilities
    :param t_in_type: type of the teacher input, 'acts' for activations/logits, 'probs' for class probabilities
    :param tau_pow: power to which the temperature is raised when used as a weighting factor in the total loss.
    :return: BxC tensor, with the gradient of the loss w.r.t. each component of the student class probs.
    """
    s_probs = opt_softmaxify(s_ins, in_type=s_in_type, t=t)
    t_probs = opt_softmaxify(t_ins, in_type=t_in_type, t=t)
    a = t_probs / s_probs
    if s_probs.shape[1] == 2:
        return (-1) * (t**tau_pow) * (a - a.flip(1))


def hkd_loss_grad_ps_v2(s_ins: Tensor, t_ins: Tensor, t, s_in_type='probs', t_in_type='probs'):
    r"""
    Computes the gradient of the HKD loss w.r.t. the student class "probabilities".
    This gradient was computed "naïvely", by applying partial derivatives rules without considering that p_S[0]+p_S[1]=1
    As such, this gradient is nonzero when it should be zero to make the loss minimal,
    however this is exactly how it is computed by PyTorch's autograd.
    :param s_ins: BxC logits/probs from the student
    :param t_ins: BxC logits/probs from the teacher
    :param t: temperature of knowledge distillation.
    :param s_in_type: type of the student input, 'acts' for activations/logits, 'probs' for class probabilities
    :param t_in_type: type of the teacher input, 'acts' for activations/logits, 'probs' for class probabilities
    :return: BxC tensor, with the gradient of the loss w.r.t. each component of the student class probs.
    """
    s_probs = opt_softmaxify(s_ins, in_type=s_in_type, t=t)
    t_probs = opt_softmaxify(t_ins, in_type=t_in_type, t=t)
    return (-1) * (t**2) * t_probs / s_probs


def hkd_loss_grad_temp(s_ins: Tensor, t_ins: Tensor, t, s_in_type='acts', t_in_type='acts'):
    s_probs = opt_softmaxify(s_ins, in_type=s_in_type, t=t)
    t_probs = opt_softmaxify(t_ins, in_type=t_in_type, t=t)
