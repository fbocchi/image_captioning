from pathlib import Path

from matplotlib import pyplot as plt


def _annotate_loss_values(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
) -> None:
    """
    Annotate the most relevant loss values on the loss-history plot.

    The following values are displayed:
        - Initial training loss
        - Initial validation loss
        - Final training loss
        - Final validation loss
        - Minimum validation loss
    """

    #
    # Initial training loss
    #
    plt.text(
        epochs[0] + 0.08,
        train_loss[0],
        f"{train_loss[0]:.3f}",
        fontsize=9,
        color="C0",
        ha="left",
        va="center",
    )

    #
    # Initial validation loss
    #
    plt.text(
        epochs[0] + 0.08,
        val_loss[0] + 0.03,
        f"{val_loss[0]:.3f}",
        fontsize=9,
        color="C1",
        ha="left",
        va="center",
    )

    #
    # Final training loss
    #
    plt.text(
        epochs[-1],
        train_loss[-1] + 0.04,
        f"{train_loss[-1]:.3f}",
        fontsize=9,
        color="C0",
        ha="center",
        va="bottom",
    )

    #
    # Final validation loss
    #
    plt.text(
        epochs[-1],
        val_loss[-1] + 0.04,
        f"{val_loss[-1]:.3f}",
        fontsize=9,
        color="C1",
        ha="center",
        va="bottom",
    )

    #
    # Minimum validation loss
    #
    best_epoch = val_loss.index(min(val_loss))

    plt.text(
        epochs[best_epoch] + 0.15,
        val_loss[best_epoch] + 0.04,
        f"{val_loss[best_epoch]:.3f}",
        fontsize=9,
        color="darkred",
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _annotate_model_checkpoint(
        epochs: list[int],
        val_loss: list[float],
) -> None:
    """
    Highlight the epoch corresponding to the best validation loss.

    This annotation marks:
        - the model saved by ModelCheckpoint;
        - the beginning of overfitting.
    """

    best_epoch = val_loss.index(min(val_loss)) + 1
    best_loss = min(val_loss)

    ###########################################################################
    # Best validation model
    ###########################################################################

    plt.scatter(
        best_epoch,
        best_loss,
        color="red",
        edgecolors="black",
        linewidth=0.8,
        s=80,
        zorder=6,
    )

    ###########################################################################
    # Beginning of overfitting
    ###########################################################################

    plt.axvline(
        best_epoch,
        color="darkred",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    ###########################################################################
    # ModelCheckpoint annotation
    ###########################################################################

    plt.text(
        best_epoch + 0.35,
        best_loss - 0.05,
        "ModelCheckpoint\n(best model)",
        fontsize=10,
        color="darkred",
        ha="left",
        va="top",
    )

    ###########################################################################
    # Overfitting annotation
    ###########################################################################

    ymin, ymax = plt.ylim()

    plt.text(
        best_epoch + 0.15,
        ymax - 0.03 * (ymax - ymin),
        "Inizio overfitting",
        fontsize=10,
        color="darkred",
        fontstyle="italic",
        ha="left",
        va="top",
    )

def _annotate_reduce_lr(
        epochs: list[int],
        learning_rate: list[float],
        reduce_lr_on_plateau_factor: float,
        reduce_lr_on_plateau_patience: int,
) -> None:
    """
    Annotate the epochs where ReduceLROnPlateau decreases
    the learning rate.
    """

    for i in range(1, len(learning_rate)):

        if learning_rate[i] >= learning_rate[i - 1]:
            continue

        plt.annotate(
            (
                "ReduceLROnPlateau\n"
                f"patience={reduce_lr_on_plateau_patience}\n"
                f"LR × {reduce_lr_on_plateau_factor:g}"
            ),
            xy=(epochs[i], learning_rate[i]),
            xycoords="data",
            xytext=(15, 20),
            textcoords="offset points",
            fontsize=10,
            color="red",
            ha="left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                shrinkA=3,
                shrinkB=3,
            ),
        )


def _plot_loss_history(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    learning_rate: list[float] | None,
    early_stopping_patience: int,
    reduce_lr_on_plateau_factor: float,
    reduce_lr_on_plateau_patience: int,
    loss_plot_out_file: Path,
) -> None:
    """
    Plot the training and validation loss histories together with the
    training events (ModelCheckpoint, ReduceLROnPlateau and EarlyStopping).
    """

    plt.figure(figsize=(8, 5))

    ###########################################################################
    # Training loss
    ###########################################################################

    plt.plot(
        epochs,
        train_loss,
        label="Training",
        linewidth=2,
        marker="o",
        markersize=5,
    )

    ###########################################################################
    # Validation loss
    ###########################################################################

    plt.plot(
        epochs,
        val_loss,
        label="Validation",
        linewidth=2,
        marker="o",
        markersize=5,
    )

    ###########################################################################
    # Display relevant loss values
    ###########################################################################

    _annotate_loss_values(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
    )

    ###########################################################################
    # ModelCheckpoint and onset of overfitting
    ###########################################################################

    _annotate_model_checkpoint(
        epochs=epochs,
        val_loss=val_loss,
    )

    ###########################################################################
    # ReduceLROnPlateau
    ###########################################################################

    if learning_rate is not None:

        _annotate_reduce_lr(
            epochs=epochs,
            learning_rate=learning_rate,
            reduce_lr_on_plateau_factor=reduce_lr_on_plateau_factor,
            reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
        )

    ###########################################################################
    # EarlyStopping
    ###########################################################################

    _annotate_early_stopping(
        epochs=epochs,
        val_loss=val_loss,
        early_stopping_patience=early_stopping_patience,
    )

    ###########################################################################
    # Axes
    ###########################################################################

    plt.title("Training e Validation Loss")

    plt.xlabel("Epoca")

    plt.ylabel("Loss")

    plt.xticks(epochs)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        loss_plot_out_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def _annotate_early_stopping(
    epochs: list[int],
    val_loss: list[float],
    early_stopping_patience: int,
) -> None:
    """
    Annotate the epoch at which training stopped because of
    EarlyStopping.
    """

    last_epoch = epochs[-1]

    plt.text(
        last_epoch - 0.35,
        val_loss[-1] + 0.10,
        (
            "EarlyStopping\n"
            f"patience={early_stopping_patience}"
        ),
        fontsize=10,
        color="black",
        ha="right",
        va="bottom",
    )


def _plot_learning_rate(
    epochs: list[int],
    learning_rate: list[float],
    reduce_lr_on_plateau_factor: float,
    reduce_lr_on_plateau_patience: int,
    output_file: Path,
) -> None:
    """
    Plot the learning-rate history and highlight the epochs where
    ReduceLROnPlateau decreased the learning rate.
    """

    plt.figure(figsize=(8, 4))

    ###########################################################################
    # Learning-rate history
    ###########################################################################

    plt.plot(
        epochs,
        learning_rate,
        label="Learning rate",
        linewidth=2,
        marker="o",
        markersize=5,
    )

    ###########################################################################
    # ReduceLROnPlateau events
    ###########################################################################

    for i in range(1, len(learning_rate)):

        if learning_rate[i] >= learning_rate[i - 1]:
            continue

        plt.scatter(
            epochs[i],
            learning_rate[i],
            color="red",
            edgecolors="black",
            linewidth=0.8,
            s=80,
            zorder=6,
        )

    ###########################################################################
    # ReduceLROnPlateau information
    ###########################################################################

    ymin, ymax = plt.ylim()

    plt.text(
        0.03,
        0.53,
        (
            "ReduceLROnPlateau\n"
            f"patience = {reduce_lr_on_plateau_patience}\n"
            f"factor = {reduce_lr_on_plateau_factor:g}"
        ),
        transform=plt.gca().transAxes,
        fontsize=10,
        color="red",
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="red",
            alpha=0.85,
            boxstyle="round,pad=0.35",
        ),
    )

    ###########################################################################
    # Axes
    ###########################################################################

    plt.title("Learning Rate History")

    plt.xlabel("Epoca")

    plt.ylabel("Learning rate")

    plt.xticks(epochs)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_history(
    history: dict[str, list[float]],
    early_stopping_patience: int,
    reduce_lr_on_plateau_factor: float,
    reduce_lr_on_plateau_patience: int,
    loss_plot_out_file: Path,
    lr_plot_out_file: Path,
) -> None:
    """
    Plot the training history.

    Two figures are produced:

        1. Training and validation loss history.
        2. Learning-rate history.
    """

    ###########################################################################
    # Create output directories
    ###########################################################################

    loss_plot_out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lr_plot_out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Training history
    ###########################################################################

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = list(
        range(
            1,
            len(train_loss) + 1,
        )
    )

    ###########################################################################
    # Loss history
    ###########################################################################

    _plot_loss_history(
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=history.get("learning_rate"),
        early_stopping_patience=early_stopping_patience,
        reduce_lr_on_plateau_factor=reduce_lr_on_plateau_factor,
        reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
        loss_plot_out_file=loss_plot_out_file,
    )

    ###########################################################################
    # Learning-rate history
    ###########################################################################

    if "learning_rate" in history:

        _plot_learning_rate(
            epochs=epochs,
            learning_rate=history["learning_rate"],
            reduce_lr_on_plateau_factor=reduce_lr_on_plateau_factor,
            reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
            output_file=lr_plot_out_file,
        )