import torch
import torch.nn as nn
from catalyst.dl.runner import SupervisedModelRunner
from catalyst.contrib.registry import Registry
from tsn import TSN, prepare_tsn_base_model


@Registry.model
def tsn(base_model, tsn_model):
    base_model = prepare_tsn_base_model(**base_model)
    net = TSN(encoder=base_model, **tsn_model)
    return net


def prepare_logdir(config):
    # CV
    model_params = config["model_params"]["tsn_model"]
    data_params = config["stages"]["data_params"]
    return f"fold_{data_params.get('in_csv_train').split('/')[-1].split('_')[2].split('.')[0]}_" \
           f"{data_params.get('uniform_time_sample')}_" \
           f"{data_params.get('n_frames')}_{data_params.get('n_segments')}_" \
           f"{model_params.get('early_consensus')}_" \
           f"{model_params.get('feature_net_skip_connection')}_" \
           f"{model_params.get('feature_net_hiddens')}_" \
           f"{','.join(model_params.get('consensus'))}_" \
           f"{model_params.get('kernel_size')}"


class ModelRunner(SupervisedModelRunner):
    @staticmethod
    def prepare_stage_model(*, model, stage, partial_bn=2, **kwargs):
        SupervisedModelRunner.prepare_stage_model(
            model=model, stage=stage, **kwargs)
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage_head_train"]:
            for param in model_.encoder.parameters():
                param.requires_grad = False
        elif stage in ["stage_full_finetune", "stage_full_train"]:
            for param in model_.encoder.parameters():
                param.requires_grad = True

            count = 0
            for m in model_.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= partial_bn:
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            pass
