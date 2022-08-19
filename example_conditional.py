# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).
import hydra
import logging
import optax
import haiku as hk
import time
import os
import matplotlib.pyplot as plt

from jaxlft import *
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)


def save_hist(name, history, params, opt_state, time_elapsed):
    params = np.asanyarray(hk.data_structures.to_mutable_dict(params), dtype=object)
    np.savez(name,
             **history,
             time=time_elapsed,
             params=params,
             opt_state=np.asanyarray(opt_state, dtype=object))


def init_live_plot(figsize=(8, 4), logit_scale=True, **kwargs):
    fig, ax_ess = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ess_line = plt.plot([0], [0.5], color='C0', label='ESS')
    plt.grid(False)
    plt.ylabel('ESS')
    if logit_scale:
        ax_ess.set_yscale('logit')
    else:
        plt.ylim(0, 1)
    ess_minmax = plt.fill_between([0], [.5], [.5], label='min/max', color='C0', alpha=0.5)

    ax_loss = ax_ess.twinx()
    loss_line = plt.plot([0], [1], color='C1', label='KL Loss')
    plt.grid(False)
    plt.ylabel('Loss')
    plt.xlabel('Steps')
    plt.legend(loc='upper right')

    lines = ess_line + ess_minmax + loss_line
    plt.legend(lines, [line.get_label() for line in lines], loc='upper center', ncol=2)

    plt.ion()
    plt.pause(0.001)
    plt.show()

    setup = dict(
        fig=fig, ax_ess=ax_ess, ax_loss=ax_loss, ess_minmax=ess_minmax,
        ess_line=ess_line, loss_line=loss_line, logit=logit_scale)

    return setup


def update_plots(history, setup, window_size=15):
    ess_line = setup['ess_line']
    ess_minmax = setup['ess_minmax']
    loss_line = setup['loss_line']
    ax_loss = setup['ax_loss']
    ax_ess = setup['ax_ess']
    fig = setup['fig']

    ess = np.mean(history['ess'], axis=1)
    ess = moving_average(ess, window=window_size)
    steps = np.arange(len(ess))
    ess_line[0].set_ydata(ess)
    ess_line[0].set_xdata(steps)
    if setup['logit'] and len(ess) > 1:
        ax_ess.relim()
        ax_ess.autoscale_view()

    ess_min = moving_average(np.min(history['ess'], axis=1), window=window_size)
    ess_max = moving_average(np.max(history['ess'], axis=1), window=window_size)
    path_y = np.concatenate([ess_max, ess_min[-1:], ess_min[::-1], ess_min[:1]])
    path_x = np.concatenate([np.arange(len(ess_min)), [len(ess_min)-1], np.arange(len(ess_max))[::-1], [0]])
    vertices = np.stack([path_x, path_y]).T
    ess_minmax.set_paths([vertices])

    loss = np.array(history['loss'])
    loss = moving_average(loss, window=window_size)
    loss_line[0].set_ydata(loss)
    loss_line[0].set_xdata(steps)
    if len(loss) > 1:
        ax_loss.relim()
        ax_loss.autoscale_view()

    fig.canvas.draw()
    plt.draw()
    plt.pause(0.001)


@hydra.main(config_path='configs', config_name='conditional', version_base=None)
def train_single(cfg):
    log.info(f'Training for sizes {cfg.lattice_size}.')
    log.info(f'Saving to {os.getcwd()}.')
    seed = cfg.seed if cfg.seed is not None else time.time_ns()
    rns = hk.PRNGSequence(seed)

    lattice_shape = (cfg.lattice_size, cfg.lattice_size)
    theory = phi4.Phi4Theory(shape=lattice_shape, m2=cfg.m2)

    opt = hydra.utils.instantiate(cfg.optimizer)
    model_def = hydra.utils.instantiate(cfg.model_def)
    model = model_def.transform(lattice_shape=lattice_shape)

    history = {
        'loss': [],
        'ess': [],
        'seed': seed,
    }

    # define training functions
    _action = jax.vmap(lambda x, lam: theory.action(x, lam=lam))

    def _loss(params, key):
        key, key_lam = jax.random.split(key)
        lam = jax.random.uniform(
            key_lam, (cfg.lam_batch_size,),
            minval=cfg.lam_range[0], maxval=cfg.lam_range[1])
        x, logq = model.sample(params, key, cfg.batch_size, lam=lam)
        x = x.reshape(cfg.lam_batch_size, -1, *x.shape[1:])
        logp = - _action(x, lam).flatten()
        dkl = reverse_dkl(logp, logq)
        return dkl, (logq, logp)
    value_and_grad = jax.value_and_grad(_loss, has_aux=True)

    @jax.jit
    def _update_step(key, params, opt_state):
        (loss, (logq, logp)), grad = value_and_grad(params, key)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        ess = jax.vmap(effective_sample_size)(
            logp.reshape(cfg.lam_batch_size, -1),
            logq.reshape(cfg.lam_batch_size, -1))
        return params, opt_state, loss, ess

    def update_step(key, params, opt_state, metrics):
        params, opt_state, loss, ess = _update_step(key, params, opt_state)
        if loss > 1e7:
            raise RuntimeError('Encountered divergent loss')
        metrics['loss'].append(loss)
        metrics['ess'].append(ess)
        return params, opt_state

    # init training
    params = model.init(next(rns), lam=cfg.lam_range[0])
    opt_state = opt.init(params)

    if cfg.live_plotting:
        plot_config = init_live_plot()
    else:
        plot_config = None

    last_save = now = start = time.time()
    log.info('Starting training...')
    epoch_step = 0
    while now < start + cfg.max_time * 60:
        params, opt_state = update_step(next(rns), params, opt_state, history)
        now = time.time()
        if cfg.save_time is not None and now > last_save + cfg.save_time * 60:
            log.info(f'Saving after {(now - start) / 60:.2f}min')
            save_hist(f'history-{lattice_shape[0]}',
                      history, params, opt_state, now - start)
            last_save = time.time()

        epoch_step += 1
        if epoch_step == cfg.epoch_size:
            if plot_config is not None:
                update_plots(history, plot_config)
            else:
                print(f'Loss: {np.mean(history["loss"][-cfg.epoch_size:])}')
                print(f'ESS: {np.mean(history["ess"][-cfg.epoch_size:])}')
            epoch_step = 0

    log.info(f'Completed training after {(now - start) / 60:.2f}min')
    save_hist(f'history-{lattice_shape[0]}',
              history, params, opt_state, now - start)


if __name__ == '__main__':
    train_single()
