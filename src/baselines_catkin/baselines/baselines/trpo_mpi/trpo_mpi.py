from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import os, os.path as osp
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import yaml
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager


# GRASPING
SIMPLE_AC = np.array([-0.2, 0.7, 0.7, 1.4])
# /GRASPING

def pretrain(pi, env):
    print("Running {} initialization episodes...".format(env.warm_init_eps), flush=True)
    n_rollouts = env.warm_init_eps
    tf_ob = U.get_placeholder_cached(name="ob")
    ob = env.reset()
    obs = np.array([ob for _ in range(n_rollouts * (env.spec.max_episode_steps + 1))])
    obs_len = 0

    graph = tf.get_default_graph()
    pdparam = graph.get_tensor_by_name("pi/pdparam:0")
    pdparam_shape = pdparam.shape[1].value
    mean, _, logstd, _ = tf.split(pdparam, [len(SIMPLE_AC), pdparam_shape // 2 - len(SIMPLE_AC),
                                            len(SIMPLE_AC), pdparam_shape // 2 - len(SIMPLE_AC)], 1)

    ac_mean = tf.constant(SIMPLE_AC, dtype=tf.float32)
    ac_logstd = tf.constant(np.array([0] * len(SIMPLE_AC)), dtype=tf.float32)

    print("Completed:", flush=True)
    for ep in range(n_rollouts):
        ob = env.reset()
        obs[obs_len] = ob
        obs_len += 1
        done = False
        while not done:
            ac, vpred = pi.act(True, ob)
            ac[:4] = SIMPLE_AC + 0.01 * np.random.randn(4)
            ac[4:] = 0
            ob, _, done, _ = env.step(ac)
            obs[obs_len] = ob
            obs_len += 1
        print(ep + 1, flush=True)

    obs = obs[:obs_len]

    with tf.variable_scope("pretrain"):
        loss = tf.nn.l2_loss(mean - ac_mean) + tf.nn.l2_loss(logstd - ac_logstd)
        opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        batch_size = 32
        num_epochs = 10
        U.get_session().run(tf.variables_initializer(set(tf.global_variables()) - U.ALREADY_INITIALIZED))
        for ep in range(num_epochs):
            for i in range(len(obs) // batch_size):
                idx = np.random.choice(len(obs), batch_size)
                U.get_session().run([opt, loss], feed_dict={tf_ob: obs[idx]})

    env.n_episodes = 0
    print("Policy initialized!", flush=True)


def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["new"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, *,
          timesteps_per_batch,  # what to train on
          log_every=None,
          log_dir=None,
          episodes_so_far=0, timesteps_so_far=0, iters_so_far=0,
          max_kl, cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          callback=None,
          **kwargs
          ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    # Target advantage function (if applicable)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g * tangent)
                    for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    # GRASPING
    saver = tf.train.Saver(var_list=U.ALREADY_INITIALIZED, max_to_keep=1)
    checkpoint = tf.train.latest_checkpoint(log_dir)
    if checkpoint:
        print("Restoring checkpoint: {}".format(checkpoint))
        saver.restore(U.get_session(), checkpoint)
    if hasattr(env, "set_actor"):
        def actor(obs):
            return pi.act(False, obs)[0]
        env.set_actor(actor)
    if not checkpoint and hasattr(env, "warm_init_eps"):
        pretrain(pi, env)
        saver.save(U.get_session(), osp.join(log_dir, "model"))
    # /GRASPING
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    tstart = time.time()

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    while True:
        if callback:
            callback(locals(), globals())
        should_break = False
        if max_timesteps and timesteps_so_far >= max_timesteps:
            should_break = True
        elif max_episodes and episodes_so_far >= max_episodes:
            should_break = True
        elif max_iters and iters_so_far >= max_iters:
            should_break = True

        if log_every and log_dir:
            if (iters_so_far + 1) % log_every == 0 or should_break:
                # To reduce space, don't specify global step.
                saver.save(U.get_session(), osp.join(log_dir, "model"))

            job_info = {'episodes_so_far': episodes_so_far,
                        'iters_so_far': iters_so_far, 'timesteps_so_far': timesteps_so_far}
            with open(osp.join(log_dir, "job_info_new.yaml"), 'w') as file:
                yaml.dump(job_info, file, default_flow_style=False)
                # Make sure write is instantaneous.
                file.flush()
                os.fsync(file)
            os.rename(osp.join(log_dir, "job_info_new.yaml"), osp.join(log_dir, "job_info.yaml"))

        if should_break:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-10)  # standardized advantage function estimate

        if hasattr(pi, "ret_rms"):
            pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(ob)  # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new()  # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)

        meanlosses = None
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather(
                    (thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        if meanlosses is not None:
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))

        logger.record_tabular("EpLenMean", np.mean(lens))
        logger.record_tabular("EpRewMean", np.mean(rews))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
