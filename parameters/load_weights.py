import numpy as np

from util import assign


def block_param(block, layer_type, tx, p_type):
    return block[layer_type][tx][p_type]


def norm_param(block, name, p_type):
    return block[name][p_type]


def block_params(block, layer_type, tx, p_type):
    return np.split(
        block_param(block, layer_type, tx, p_type),
        3,
        axis=-1
    )


def block_weights(block, layer_type, tx):
    return block_params(block, layer_type, tx, "w")


def block_weight(block, layer_type, tx):
    return block_param(block, layer_type, tx, "w")


def block_biases(block, layer_type, tx):
    return block_params(block, layer_type, tx, "b")


def block_bias(block, layer_type, tx):
    return block_param(block, layer_type, tx, "b")


def load_bias(target, bias):
    target.bias = assign(target.bias, bias)


def load_weight(target, weight):
    target.weight = assign(target.weight, weight)


def load_scale(target, scale):
    target.scale = assign(target.scale, scale)


def load_shift(target, shift):
    target.shift = assign(target.shift, shift)


def load_weights(block, p_block, layer_type, tx):
    q_w, k_w, v_w = block_weights(p_block, layer_type, tx)
    load_weight(block.att.W_query, q_w.T)
    load_weight(block.att.W_key, k_w.T)
    load_weight(block.att.W_value, v_w.T)


def load_biases(block, p_block, layer_type, tx):
    q_b, k_b, v_b = block_biases(p_block, layer_type, tx)
    load_bias(block.att.W_query, q_b)
    load_bias(block.att.W_key, k_b)
    load_bias(block.att.W_value, v_b)


def load_norm(target, p_block, name):
    load_scale(target, norm_param(p_block, name, "g"))
    load_shift(target, norm_param(p_block, name, "b"))


def load_block(block, p_block):
    load_weights(block, p_block, "attn", "c_attn")
    load_biases(block, p_block, "attn", "c_attn")
    load_layer(block.att.out_proj, p_block, "attn", "c_proj")
    load_layer(block.ff.layers[0], p_block, "mlp", "c_fc")
    load_layer(block.ff.layers[2], p_block, "mlp", "c_proj")
    load_norm(block.norm1, p_block, "ln_1")
    load_norm(block.norm2, p_block, "ln_2")


def load_layer(target, p_block, layer_type, tx):
    load_bias(
        target,
        block_bias(p_block, layer_type, tx))
    load_weight(
        target,
        block_weight(p_block, layer_type, tx).T
    )


def load_gpt_weights(gpt, params):
    load_weight(gpt.pos_emb, params.wpe)
    load_weight(gpt.tok_emb, params.wte)

    for b in range(len(params.blocks)):
        load_block(gpt.trf_blocks[b], params.blocks[b])

    load_scale(gpt.final_norm, params.g)
    load_shift(gpt.final_norm, params.b)
    load_weight(gpt.out_head, params.wte)
