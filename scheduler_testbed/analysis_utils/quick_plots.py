"""Quick plots for analysis."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
import itertools

# Third Party Imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import abs
from ray.tune.result_grid import ResultGrid


# %% Utility function
def scale(x, norm_factor):
    """Used to scale x."""
    return (x - norm_factor) / abs(norm_factor)


# %% Plot Learning Curves
def plotLearningCurves(
    result_grids: list[ResultGrid],
    plot_min_max: bool = False,
    reward_as_percent: bool = False,
    match_linestyles: bool = True,
    **kwargs,
) -> Figure:
    """Plot learning curve and KL divergence for a list of result grids."""
    fig, axs = plt.subplots(2, sharex=True)

    # Match line styles for same series in reward and KL subplots
    if match_linestyles is True:
        marker_iter = itertools.cycle((",", "*", ".", "|", "1", "2"))
        linestyle_iter = itertools.cycle(("-", "--", ":", "-."))
    else:
        marker_iter = None
        linestyle_iter = None

    # Plot mean episode reward for top subplot
    axs[0] = plotEpisodeReward(
        result_grids,
        plot_min_max,
        reward_as_percent,
        ax=axs[0],
        marker_iter=marker_iter,
        linestyle_iter=linestyle_iter,
        **kwargs,
    )

    # Plot KL divergence on bottom subplot
    axs[1] = plotKLDivergence(
        result_grids,
        ax=axs[1],
        marker_iter=marker_iter,
        linestyle_iter=linestyle_iter,
        **kwargs,
    )

    return fig


def plotEpisodeReward(
    result_grids: list[ResultGrid],
    plot_min_max: bool = False,
    reward_as_percent: bool = False,
    ax=None,
    marker_iter: itertools.iterable = None,
    linestyle_iter: itertools.iterable = None,
    **kwargs,
):
    """Plot episode reward for a list of result grids."""
    if ax is None:
        ax = plt.gca()

    # Normalization factors divide magnitude; used for percentage plots.
    # Each entry of norm_factors is (key, value) = (trial id, episode_reward_mean[1]).
    # Choose 1st (rather than 0th) entry of episode_reward_mean b/c the 0th entry
    # is always NaN. Trials that errored do not get a (key, value) entry. If
    # reward_as_percent is False, then just set all norm_factors to 1.
    norm_factors = [{} for i in range(len(result_grids))]
    if reward_as_percent is True:
        for i, grid in enumerate(result_grids):
            trial_ids = grid.get_dataframe()["trial_id"]
            norm_factors[i] = dict.fromkeys(list(trial_ids))
            for result in grid:
                if result.checkpoint is None:
                    # Skip jth loop if result doesn't have a checkpoint(for errored
                    # trials)
                    continue
                trial_id = result.metrics_dataframe["trial_id"][0]
                norm_factors[i][trial_id] = result.metrics_dataframe[
                    "episode_reward_mean"
                ][1]
    else:
        for i, grid in enumerate(result_grids):
            trial_ids = grid.get_dataframe()["trial_id"]
            norm_factors[i] = dict.fromkeys(list(trial_ids), 1)

    # Loop through result grids and plot each trial as a separate line. Optionally
    # include min/max shaded areas in plot. Optionally plot all values as percentage
    # of initial value. Skip plotting trials that errored (no data to plot). Optionally
    # set marker and linestyle of plot to next values in iterator.
    for i, grid in enumerate(result_grids):
        for result in grid:
            if result.checkpoint is None:
                # Skip jth loop if result doesn't have a checkpoint(for errored trials)
                continue

            trial_id = result.metrics_dataframe["trial_id"][0]

            # Manually set marker and linestyle if iterators are provided.
            if marker_iter is not None:
                kwargs["marker"] = next(marker_iter)
            if linestyle_iter is not None:
                kwargs["linestyle"] = next(linestyle_iter)

            # Scale values to plot. If reward_as_percent == False, then this step
            # does nothing.
            scaled_mean = scale(
                result.metrics_dataframe["episode_reward_mean"],
                norm_factors[i][trial_id],
            )
            (line,) = ax.plot(
                result.metrics_dataframe["training_iteration"],
                scaled_mean,
                label=trial_id,
                **kwargs,
            )

            # Optionally plot min/max values as shaded regions between min/max
            # and mean. Color of shaded regions matches line color, but with a
            # partial transparency.
            if plot_min_max is True:
                scaled_min = scale(
                    result.metrics_dataframe["episode_reward_min"],
                    norm_factors[i][trial_id],
                )
                scaled_max = scale(
                    result.metrics_dataframe["episode_reward_max"],
                    norm_factors[i][trial_id],
                )
                line_color = ax.lines[-1].get_color()
                ax.fill_between(
                    result.metrics_dataframe["training_iteration"],
                    scaled_min,
                    scaled_mean,
                    color=line_color,
                    alpha=0.25,
                )
                ax.fill_between(
                    result.metrics_dataframe["training_iteration"],
                    scaled_max,
                    scaled_mean,
                    color=line_color,
                    alpha=0.25,
                )

    # Set y-axis label depending on percentage or absolute reward
    if reward_as_percent is False:
        ax.set_ylabel("episode_reward_mean")
    else:
        ax.set_ylabel("pct-change episode_reward_mean")

    ax.set_xlabel("training_iteration")
    ax.legend()
    return ax


def plotKLDivergence(
    result_grids: list[ResultGrid],
    ax=None,
    marker_iter: itertools.iterable = None,
    linestyle_iter: itertools.iterable = None,
    **kwargs,
):
    """Plot KL Divergence for a list of result_grids."""
    if ax is None:
        ax = plt.gca()

    for grid in result_grids:
        # labels = list(grid.get_dataframe().get("trial_id"))
        for result in grid:
            if result.checkpoint is None:
                # Skip jth loop if result doesn't have a checkpoint(for errored
                # trials)
                continue

            label = result.metrics_dataframe["trial_id"][0]

            if marker_iter is not None:
                kwargs["marker"] = next(marker_iter)
            if linestyle_iter is not None:
                kwargs["linestyle"] = next(linestyle_iter)

            result.metrics_dataframe.plot(
                "training_iteration",
                "info/learner/default_policy/learner_stats/kl",
                ax=ax,
                label=label,
                **kwargs,
            )

    ax.set_ylabel("KL Divergence")

    return ax
