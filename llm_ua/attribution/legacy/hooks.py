from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseRetainGradHook:

    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        # cache stores the gradient (already copied to cpu)
        self.cache: Optional[torch.Tensor] = None
        self.handle = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def _custom_retain_grad(self, output: torch.Tensor) -> torch.Tensor:
        # if output is a tuple, e.g. output from transformer encoder layer, it is assumed that the first element
        # is the hidden_states
        if isinstance(output, tuple):
            targeted_output = output[0]
        elif isinstance(output, torch.Tensor):
            targeted_output = output
        else:
            raise TypeError(f'Unsupported type for _custom_retain_grad: {type(output)}')

        # Instead of retain_grad, use a backward hook
        def backward_hook(grad):
            self.cache = grad.detach().cpu()

        # Attach the backward hook
        self.handle = targeted_output.register_hook(backward_hook)

        return output

    def reset_cache(self):
        if self.cache is not None:
            self.handle.remove()
            self.handle = None
            self.cache = None


class RetainGradWithKwargs(BaseRetainGradHook):

    def __init__(self, layer_name: str) -> None:
        super().__init__(layer_name=layer_name)

    def __call__(self, module, args, kwargs, output):
        return self._custom_retain_grad(output)


class RetainGradWithoutKwargs(BaseRetainGradHook):

    def __init__(self, layer_name: str) -> None:
        super().__init__(layer_name=layer_name)

    def __call__(self, module, args, output):
        return self._custom_retain_grad(output)


class HookManager:

    def __init__(self, ) -> None:
        self.hooks = []
        self.handles = []

    def register_hooks(self, model: nn.Module, layer_groups: List[Dict]) -> nn.Module:
        for group in layer_groups:
            layer_name_template = group['name_template']
            layer_indices = group['indices']
            with_kwargs = group['with_kwargs']

            if len(layer_indices) == 0:
                # layer_name_template represents a unique layer name and does not contain any index. E.g. embed_tokens.
                layer_name = layer_name_template
                self._register_single_hook(model, layer_name, with_kwargs=with_kwargs)
            else:
                # layer_name_template can be formatted with layer_index. E.g. layers.1.self.attn.
                for layer_index in layer_indices:
                    layer_name = layer_name_template.format(layer_index)
                    self._register_single_hook(model, layer_name, with_kwargs=with_kwargs)

        return model

    def _register_single_hook(self, model: nn.Module, layer_name: str, with_kwargs: bool) -> None:
        layer = model.get_submodule(layer_name)
        if with_kwargs:
            hook = RetainGradWithKwargs(layer_name=layer_name)
            self.handles.append(layer.register_forward_hook(hook=hook, with_kwargs=True))
        else:
            hook = RetainGradWithoutKwargs(layer_name=layer_name)
            self.handles.append(layer.register_forward_hook(hook=hook, with_kwargs=False))
        self.hooks.append(hook)

    def read_out_gradients(self) -> Dict[str, torch.Tensor]:
        gradient_dict: Dict[str, torch.Tensor] = dict()
        for hook in self.hooks:
            layer_name = hook.layer_name
            grad = hook.cache
            gradient_dict.update({layer_name: grad})
        return gradient_dict

    def clear_hooks(self):
        self.reset_cache()
        self.hooks.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset_cache(self):
        for hook in self.hooks:
            hook.reset_cache()
