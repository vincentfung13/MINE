import torch
import os
import subprocess


def disparity_normalization_vis(disparity):
    """

    :param disparity: Bx1xHxW, pytorch tensor of float32
    :return:
    """
    assert len(disparity.size()) == 4 and disparity.size(1) == 1
    disp_min = torch.amin(disparity, (1, 2, 3), keepdim=True)
    disp_max = torch.amax(disparity, (1, 2, 3), keepdim=True)
    disparity_syn_scaled = (disparity - disp_min) / (disp_max - disp_min)
    disparity_syn_scaled = torch.clip(disparity_syn_scaled, 0.0, 1.0)
    return disparity_syn_scaled


def run_shell_cmd(args_list, logger):
    """
    run linux commands
    """
    if logger:
        logger.info("Running system command: {0}".format(" ".join(args_list)))
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err


def run_shell_cmd_shell(cmd, logger):
    logger.info("Running system command: {0}".format(cmd))
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err


def restore_model(model_path, backbone, decoder, optimizer, logger):
    if model_path is None:
        if logger:
            logger.info("Not using pre-trained model...")
        return

    assert os.path.exists(model_path), "Model %s does not exist!"
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cpu())

    for key, model in [("backbone", backbone), ("decoder", decoder), ("optimizer", optimizer)]:
        if key not in state_dict:
            continue

        _state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v
                       for k, v in state_dict[key].items()}

        # Check if there is key mismatch:
        missing_in_model = set(_state_dict.keys()) - set(model.state_dict().keys())
        missing_in_ckp = set(model.state_dict().keys()) - set(_state_dict.keys())

        if logger:
            logger.info("[MODEL_RESTORE] missing keys in %s checkpoint: %s" % (key, missing_in_ckp))
            logger.info("[MODEL_RESTORE] missing keys in %s model: %s" % (key, missing_in_model))

        if key != "optimizer":
            model.load_state_dict(_state_dict, strict=False)
        else:
            model.load_state_dict(_state_dict)


def linspace_batch(start, end, steps, dtype=None, device=None):
    """
    x0 ... x ... x1
    y0 ... y ... y1
    (x-x0) / (x1-x0) = (y-y0) / (y1-y0)
    :param start: B
    :param end: B
    :param steps: int
    :param dtype:
    :param device:
    :return:
    """
    assert len(start.size()) == 1 and start.size() == end.size()
    assert isinstance(steps, int)

    B, S = start.size(0), steps
    x = torch.linspace(start[0], end[0], steps=S, dtype=dtype, device=device)  # S
    x = x.unsqueeze(0).repeat(B, 1)  # BxS

    x0 = torch.full((B, S), fill_value=start[0], dtype=dtype, device=device)
    x1 = torch.full((B, S), fill_value=end[0], dtype=dtype, device=device)
    linear_arr = (end-start).unsqueeze(1) * (x - x0) / (x1 - x0) + start.unsqueeze(1)

    return linear_arr


def inverse(matrices):
    """
    torch.inverse() sometimes produces outputs with nan the when batch size is 2.
    Ref https://github.com/pytorch/pytorch/issues/47272
    this function keeps inversing the matrix until successful or maximum tries is reached
    :param matrices Bx3x3
    """
    inverse = None
    max_tries = 5
    while (inverse is None) or (torch.isnan(inverse)).any():
        torch.cuda.synchronize()
        inverse = torch.inverse(matrices)

        # Break out of the loop when the inverse is successful or there"re no more tries
        max_tries -= 1
        if max_tries == 0:
            break

    # Raise an Exception if the inverse contains nan
    if (torch.isnan(inverse)).any():
        raise Exception("Matrix inverse contains nan!")
    return inverse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Embedder(object):
    # Positional encoding (section 5.1)
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    embed_kwargs = {
                "include_input": True,
                "input_dims": 1,
                "max_freq_log2": multires - 1,
                "num_freqs": multires,
                "log_sampling": True,
                "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


def test_linspace_batch():
    B, S = 8, 10
    start = torch.arange(B)*10 + 10
    end = torch.arange(B) + 1

    batch_linspace = linspace_batch(start, end, S)

    single_linspace_list = []
    for i in range(B):
        single_linspace_list.append(torch.linspace(start[i], end[i], S))
    assemble_linspace = torch.stack(single_linspace_list, dim=0)

    print(batch_linspace - assemble_linspace)


if __name__ == "__main__":
    embedder, out_dim = get_embedder(20)
    input_t = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 4]])
    print(embedder(input_t).size())
