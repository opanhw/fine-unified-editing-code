from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .fine_hparams import FINEHyperParams
import torch.nn.functional as F
from .repr_tools import get_words_idxs_in_templates


CONTEXT_TEMPLATES_CACHE = None



def apply_template(text):
    return f"<s>[INST] {text} [/INST]"



def apply_fine_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: List[Dict],
    hparams: FINEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs,
):
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    # torch.manual_seed(42)

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    deltas, model_top_neurons = [], []

    for idx, single_request in enumerate(request):

        single_deltas, single_model_top_neurons, _weights_copy = execute_fine(model, tok, single_request, hparams, idx, **kwargs)

        deltas.append(single_deltas)
        model_top_neurons.append(single_model_top_neurons)

        if return_orig_weights:
            weights_copy.update(_weights_copy)

    deltas = torch.cat(deltas, dim=0)
    model_top_neurons = torch.cat(model_top_neurons, dim=0)

    if (kwargs.get('find_only_first') is not None and kwargs['find_only_first']) or (kwargs.get('find_only_second') is not None and kwargs['find_only_second']):
        return model, weights_copy

    # modify model state dict
    state_dict = model.state_dict()

    for j, neuron in enumerate(model_top_neurons):
        delta = deltas[j]
        w = state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'].T
        w[neuron[1]] += delta.to(w.device)
        state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'] = w.T

    # load new state dict
    model.load_state_dict(state_dict)

    if return_orig_weights:
        weights_copy = _weights_copy

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def locate_neurons(
    model,
    tok,
    request,
    hparams,
):
    inputs = tok([request["target_new"].strip(), request["ground_truth"]], return_tensors="pt", padding=True, truncation=True).to(model.device)

    act = [[] for _ in range(model.config.num_hidden_layers)]

    def forward_hook(n):
        def fn(_, input, output):
            # print(f'output.shape:{output.shape}')
            act[n].append(output.detach().half().cpu())

        return fn

    handle_act = [model.model.layers[n].mlp.act_fn.register_forward_hook(forward_hook(n)) for n in
                    range(model.config.num_hidden_layers)]


    with torch.no_grad():
        outputs = model(**inputs)
    
    activations = [act[n][0] for n in range(model.config.num_hidden_layers)]

    for h in handle_act:
        h.remove()

    activations = torch.stack(activations).permute(1, 2, 0, 3)
    
    dist = ((activations[0] - activations[1]) ** 2).mean(0)

    if hparams.layer is not None:
        _, top_index = torch.topk(dist[hparams.layer], hparams.safety_neuron_num)
        top_pos = [(hparams.layer, x.item()) for x in top_index]
    else:
        if hparams.last_layer:
            dist = dist[:-hparams.last_layer]
        _, top_index = torch.topk(dist.flatten(), hparams.safety_neuron_num)
        top_pos = [(x.item() // dist.shape[1], x.item() % dist.shape[1]) for x in top_index]

    return torch.tensor(top_pos)


def _locate_neurons(
    model,
    tok,
    request,
    hparams,
    target_ids
):
    input_ids = torch.cat([tok.encode(request['prompt'], add_special_tokens=True, return_tensors="pt"), target_ids], dim=-1).cuda()

    target_len = target_ids.shape[-1]

    act = [[] for _ in range(model.config.num_hidden_layers)]

    def forward_hook(n):
        def fn(_, input, output):
            # print(f'output.shape:{output.shape}')
            act[n].append(output.detach().half().cpu())

        return fn

    handle_act = [model.model.layers[n].mlp.act_fn.register_forward_hook(forward_hook(n)) for n in
                    range(model.config.num_hidden_layers)]


    with torch.no_grad():
        model_outputs = model(
            input_ids=input_ids,
        )
    
    activations = [act[n][0] for n in range(model.config.num_hidden_layers)]

    for h in handle_act:
        h.remove()

    original_distribution = torch.cat([model_outputs.logits[0, :-target_len - 1]], dim=0)
    original_distribution_last = model_outputs.logits[0, -1:]

    model_outputs_ids = model_outputs.logits.argmax(-1).cpu()
    original_output_ids = model_outputs_ids[0, -target_len - 1:-1]
    activations = torch.stack(activations).permute(1, 2, 0, 3)[0, -target_len - 1:-1]

    # print(activations.shape)

    top_neurons = []

    for j, id in enumerate(original_output_ids):
        scores = torch.zeros((activations.shape[1] - hparams.last_layer, activations.shape[2])).half().cuda()
        unembedding = model.lm_head.weight.data.T[:, id].half()
        for layer in range(activations.shape[1] - hparams.last_layer):
            act = activations[j, layer].cuda()

            if 'gpt-j' in hparams.model_name:
                down_proj = model.transformer.h[layer].mlp.fc_out.weight.data.T.half()
            else:
                down_proj = model.model.layers[layer].mlp.down_proj.weight.data.T.half()
            

            prob = act * ((down_proj.to(act.device) @ unembedding.to(act.device)).flatten())
            # prob = act
            scores[layer] += prob.flatten()

        if hparams.layer is not None:
            _, top_index = torch.topk(activations[hparams.layer], hparams.neuron_num)
            top_pos = [(hparams.layer, x.item()) for x in top_index]
        else:
            _, top_index = torch.topk(scores.flatten(), hparams.neuron_num)
            top_pos = [[x.item() // scores.shape[1], x.item() % scores.shape[1]] for x in top_index]

        top_neurons.append(top_pos)


    return torch.tensor(top_neurons), original_distribution


def execute_fine(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: FINEHyperParams,
    request_idx: int,
    **kwargs,
):
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)

    # request["target_new"] = "I'm sorry, but"


    if request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    print(
        f"Executing FINE algorithm for the update: "
        f"[{request['prompt']}] -> [{request['target_new']}]"
    )

    inputs = tok(request["prompt"] + request["target_new"], return_tensors="pt").to(model.device)
    if "llama-2" in hparams.model_name.lower():
        target_ids = tok.encode(request['target_new'].strip(), add_special_tokens=False, return_tensors="pt")
    else:
        target_ids = tok.encode(request['target_new'], add_special_tokens=False, return_tensors="pt")
    
    target_len = target_ids.shape[-1]

    print(f"Finding neurons...")

    model.eval()
    for param in model.named_parameters():
        param[1].requires_grad = False

    # hparams.random = True
    
    if hparams.random:
        print("Randomly selected top_neurons")
        first_top_neurons = torch.zeros(hparams.safety_neuron_num, 2, dtype=torch.int16)
        for j, y in enumerate(first_top_neurons):
            if hparams.layer is None:
                first_top_neurons[j, 0] = torch.randint(0, model.config.num_hidden_layers - hparams.last_layer, (1, ))[0]
            else:
                first_top_neurons[j, 0] = hparams.layer
            first_top_neurons[j, 1] = torch.randint(0, model.config.intermediate_size, (1, ))[0]
    else:
        first_top_neurons = locate_neurons(model, tok, request, hparams)

    # hparams.random = False
    
    print(first_top_neurons.shape)

    first_top_neurons = first_top_neurons.reshape(-1, 2)

    if kwargs.get('find_only_first') is not None and kwargs['find_only_first']:
        torch.save(first_top_neurons, kwargs['find_file_path'])
        return torch.tensor([]), torch.tensor([]), {}

    first_neurons_dict = {}

    for i, neuron in enumerate(first_top_neurons):
        neuron = neuron.tolist()
        if neuron[0] not in first_neurons_dict:
            first_neurons_dict[neuron[0]] = {"loc": [neuron[1]], "idx": [i]}
        else:
            first_neurons_dict[neuron[0]]['loc'].append(neuron[1])
            first_neurons_dict[neuron[0]]['idx'].append(i)

    
    
    deltas = (torch.zeros(first_top_neurons.shape[0], model.config.hidden_size, dtype=torch.bfloat16)).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([deltas], lr=hparams.lr)

    def forward_hook(n):
        def fn(module, input, output):
            if n in first_neurons_dict:
                neuron_loc = first_neurons_dict[n]['loc']
                neuron_idx = first_neurons_dict[n]['idx']
                output_delta = input[0][:, :, neuron_loc].to(deltas.dtype) @ deltas[neuron_idx].requires_grad_(True)

                return output[0] + output_delta.to(input[0].dtype)

        return fn

    if model.config.model_type == "llama" or model.config.model_type == "qwen2" or model.config.model_type == "qwen3":
        handle = [model.model.layers[n].mlp.down_proj.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]
    elif model.config.model_type == "gptj":
        handle = [model.transformer.h[n].mlp.fc_out.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]


    print(sum([torch.cuda.max_memory_allocated(f"cuda:{i}") / 1024 ** 3 for i in range(torch.cuda.device_count())]))
    print("Start editing...")

    

    for epoch in range(hparams.epochs):
        model_outputs = model.forward(**inputs)

        output = model_outputs.logits[0, -target_len-1:-1].float()

        nll_loss = F.nll_loss(F.log_softmax(output, dim=-1), target_ids[0, :].to(output.device), reduction="none").max()


        norm_loss = torch.norm(deltas, p=2) * hparams.beta
        loss = nll_loss + norm_loss

        if (epoch + 1) % 1 == 0:
            print(
                f"Request {request_idx} "
                f"Epoch {epoch + 1} Loss: {loss.item():.4f} NLL: {nll_loss.item():.4f} Norm: {norm_loss.item():.4f} "
            )
        
        if hparams.early_stop_prob is not None and nll_loss < -np.log(hparams.early_stop_prob) or np.isnan(nll_loss.detach().cpu()):
            break

        loss.backward()
        # print(deltas.grad)
        optimizer.step()
        optimizer.zero_grad()

    for h in handle:
        h.remove()
    
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in list(set(first_top_neurons[:, 0].flatten().tolist()))
    }

    state_dict = model.state_dict()

    for j, neuron in enumerate(first_top_neurons):
        delta = deltas[j]
        w = state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'].T
        w[neuron[1]] += delta.to(w.device)
        state_dict[f'{hparams.rewrite_module_tmp.format(neuron[0])}.weight'] = w.T

    # load new state dict
    model.load_state_dict(state_dict)


    target_new = tok.batch_decode(tok.encode(request['target_new'], add_special_tokens=False, return_tensors="pt")[:, :10], skip_special_tokens=False)[0]
    
    inputs = tok(request["prompt"] + target_new, return_tensors="pt").to(model.device)
    if "llama-2" in hparams.model_name.lower():
        target_ids = tok.encode(target_new.strip(), add_special_tokens=False, return_tensors="pt")
    else:
        target_ids = tok.encode(target_new, add_special_tokens=False, return_tensors="pt")
    
    target_len = target_ids.shape[-1]

    # hparams.random = True
    
    if hparams.random:
        print("Randomly selected top_neurons")
        second_top_neurons = torch.zeros(10, hparams.neuron_num, 2, dtype=torch.int16)
        for i, x in enumerate(second_top_neurons):
            for j, y in enumerate(x):
                if hparams.layer is None:
                    second_top_neurons[i, j, 0] = torch.randint(0, model.config.num_hidden_layers - hparams.last_layer, (1, ))[0]
                else:
                    second_top_neurons[i, j, 0] = hparams.layer
                second_top_neurons[i, j, 1] = torch.randint(0, model.config.intermediate_size, (1, ))[0]
        _ , original_distribution = _locate_neurons(model, tok, request, hparams, target_ids)
    else:
        second_top_neurons, original_distribution = _locate_neurons(model, tok, request, hparams, target_ids)

    # hparams.random = False
    
    print(second_top_neurons.shape)

    second_top_neurons = second_top_neurons.reshape(-1, 2)


    if kwargs.get('find_only_second') is not None and kwargs['find_only_second']:
        torch.save(second_top_neurons, kwargs['find_file_path'])
        return torch.tensor([]), torch.tensor([]), {}
    
    second_neurons_dict = {}

    for i, neuron in enumerate(second_top_neurons):
        neuron = neuron.tolist()
        if neuron[0] not in second_neurons_dict:
            second_neurons_dict[neuron[0]] = {"loc": [neuron[1]], "idx": [i]}
        else:
            second_neurons_dict[neuron[0]]['loc'].append(neuron[1])
            second_neurons_dict[neuron[0]]['idx'].append(i)

    deltas = (torch.zeros(second_top_neurons.shape[0], model.config.hidden_size, dtype=torch.bfloat16)).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([deltas], lr=hparams.lr)

    def forward_hook(n):
        def fn(module, input, output):
            if n in second_neurons_dict:
                neuron_loc = second_neurons_dict[n]['loc']
                neuron_idx = second_neurons_dict[n]['idx']
                output_delta = input[0][:, :, neuron_loc].to(deltas.dtype) @ deltas[neuron_idx].requires_grad_(True)

                return output[0] + output_delta.to(input[0].dtype)

        return fn

    if model.config.model_type == "llama" or model.config.model_type == "qwen2" or model.config.model_type == "qwen3":
        handle = [model.model.layers[n].mlp.down_proj.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]
    elif model.config.model_type == "gptj":
        handle = [model.transformer.h[n].mlp.fc_out.register_forward_hook(forward_hook(n)) for n in
                  range(model.config.num_hidden_layers)]


    print(sum([torch.cuda.max_memory_allocated(f"cuda:{i}") / 1024 ** 3 for i in range(torch.cuda.device_count())]))
    print("Start editing...")

    nan_flag = False

    for epoch in range(hparams.epochs):
        model_outputs = model.forward(**inputs)

        output = model_outputs.logits[0, -target_len-1:-1].float()

        nll_loss = F.nll_loss(F.log_softmax(output, dim=-1), target_ids[0, :].to(output.device), reduction="none").max()

        kl_loss = F.kl_div(F.log_softmax(model_outputs.logits[0, :-target_len - 1], dim=-1),
                           F.softmax(original_distribution, dim=-1), reduction='batchmean') * hparams.theta
        
        penalty_loss = (-torch.log(1 - F.softmax(model_outputs.logits[0, -1].float(), dim=-1)[target_ids[0, :]])).max() * hparams.gamma

        norm_loss = torch.norm(deltas, p=2) * hparams.beta
        loss = nll_loss + norm_loss + kl_loss + penalty_loss


        if (epoch + 1) % 1 == 0:
            print(
                f"Request {request_idx} "
                f"Epoch {epoch + 1} Loss: {loss.item():.4f} NLL: {nll_loss.item():.4f} Norm: {norm_loss.item():.4f} KL: {kl_loss.item():.4f} Pen: {penalty_loss.item():.4f} "
                f"avg prob of [{target_new}] "
                f"{torch.gather(F.softmax(output, dim=-1), -1, target_ids[0, :].unsqueeze(-1).to(output.device)).min().item():.8f} "
                f"output id: {model_outputs.logits[0, -target_len-1:-1].float().argmax(-1).tolist()} "
                f"{F.softmax(model_outputs.logits[0, -1].float(), dim=-1)[target_ids[0, :]].tolist()}"
            )
        
        if hparams.early_stop_prob is not None and nll_loss < -np.log(hparams.early_stop_prob):
            break
            
        if np.isnan(nll_loss.detach().cpu()):
            nan_flag = True
            break

        loss.backward()
        # print(deltas.grad)
        optimizer.step()
        optimizer.zero_grad()

    for h in handle:
        h.remove()
    
    if nan_flag:
        deltas = torch.zeros(second_top_neurons.shape[0], model.config.hidden_size, dtype=torch.bfloat16).cuda()
    else:
        # Retrieve weights that user desires to change
        weights.update({
            f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in list(set(second_top_neurons[:, 0].flatten().tolist())) if f"{hparams.rewrite_module_tmp.format(layer)}.weight" not in weights
        })

    print(weights.keys())

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    return deltas.detach(), second_top_neurons, weights_copy



def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
